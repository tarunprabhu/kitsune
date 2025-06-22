//===- ResolveDeviceFuncsInEmbBC.cpp - Resolve device functions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The embedded bitcode may contain calls to library functions for which
// device-specific implementations exist. This resolves the calls to such
// functions in the embedded bitcode to use the device-specific implementations.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/ResolveDeviceFuncs.h"
#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/Transforms/EmbBCPassUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

#define DEBUG_TYPE "resolve-device-funcs"

using namespace llvm;

namespace {

/// Implementation class to resolve calls to device functions in kernel modules.
/// This specifically replaces calls to functions known to have a
/// device-specific implementations in libDevice modules.
class ResolveDeviceFuncs {
private:
  TTID tt;
  const TapirTargetOptions &tto;

private:
  Function *resolveCallee(Function *f, bool fast, Module &libDeviceM) {
    std::string devFnName = getDeviceFunc(f->getName(), fast);
    if (not devFnName.empty()) {
      if (Function *devFn = libDeviceM.getFunction(devFnName)) {
        LLVM_DEBUG(dbgs() << "resolve-device-funcs: mapped function '"
                          << f->getName() << "' to '" << devFnName << "'\n");
        // If the device function has already been declared in the kernel
        // module, create a declaration for the device function with the correct
        // attributes. We may need to fix the linkage type here because some
        // linkages are not allowed on declarations. The link-device-bitcode
        // pass will provide definitions for these functions.
        Module *m = f->getParent();
        Function *fdecl = m->getFunction(devFnName);
        if (not fdecl) {
          FunctionType *ftype = devFn->getFunctionType();
          GlobalValue::LinkageTypes linkage = devFn->getLinkage();
          if (not GlobalValue::isValidDeclarationLinkage(linkage))
            linkage = GlobalValue::ExternalLinkage;

          fdecl = Function::Create(ftype, linkage, devFnName, m);
          fdecl->setAttributes(devFn->getAttributes());
          fdecl->setCallingConv(devFn->getCallingConv());
        }
        return fdecl;
      }
      // If a device function was not found, we can't resolve anything. This is
      // an error, but we don't deal with it here.
      LLVM_DEBUG(dbgs() << "resolve-device-funcs: WARNING: Mapped function '"
                        << devFnName << "' not in libdevice module" << "\n");
    }

    // Returning null to the caller indicates that the callee was not found in
    // the libdevice module and does not need to be replaced. This is not
    // necessarily an error. For instance, if the function is a target intrinsic
    // or is already a function from the libdevice module, there is no need to
    // replace it. However, if it is a function that should have been handled,
    // this will result in an error later in the compilation process.
    LLVM_DEBUG(dbgs() << "resolve-device-funcs: Not resolving '" << f->getName()
                      << "'\n");
    return nullptr;
  }

  bool resolveCallees(Function &f, Module &libDeviceM) {
    LLVM_DEBUG(dbgs() << "resolve-device-funcs: In '" << f.getName() << "'\n");

    bool changed = false;
    for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i) {
      if (auto *call = dyn_cast<CallBase>(&*i)) {
        bool enableFast = false;
        if (auto *fpo = dyn_cast<FPMathOperator>(call))
          enableFast = fpo->isFast();

        if (Function *cf = call->getCalledFunction()) {
          if (cf->isDeclaration()) {
            if (Function *df = resolveCallee(cf, enableFast, libDeviceM)) {
              call->setCalledFunction(df);
              changed |= true;
            }
          }
        }
      }
    }

    return changed;
  }

protected:
  ResolveDeviceFuncs(TTID tt, const TapirTargetOptions &tto)
      : tt(tt), tto(tto) {}

  virtual std::string getDeviceFunc(StringRef f, bool fast) = 0;

public:
  virtual ~ResolveDeviceFuncs() = default;

  bool run(Module &m) {
    LLVMContext &ctx = m.getContext();
    std::unique_ptr<Module> libDeviceM = getLibDeviceModule(tt, tto, ctx);

    bool changed = false;
    for (Function &f : m.functions())
      if (f.size())
        changed |= resolveCallees(f, *libDeviceM);
    return changed;
  }
};

/// Resolve device functions for cuda.
class ResolveDeviceFuncsCuda : public ResolveDeviceFuncs {
private:
  static const StringMap<StringRef> devFuncs;

protected:
  std::string getDeviceFunc(StringRef f, bool fast) override final {
    if (devFuncs.find(f) == devFuncs.end())
      return "";

    StringRef pfx = fast ? "__nv_fast_" : "__nv_";
    return join_items("", pfx, devFuncs.at(f));
  }

public:
  ResolveDeviceFuncsCuda(const TapirTargetOptions &tto)
      : ResolveDeviceFuncs(TTID::Cuda, tto) {}
};

const StringMap<StringRef> ResolveDeviceFuncsCuda::devFuncs = {
    {"acos", "acos"},
    {"acosf", "acosf"},
    {"acosh", "acosh"},
    {"acoshf", "acoshf"},
    {"asin", "asin"},
    {"asinf", "asinf"},
    {"asinh", "asinh"},
    {"asinhf", "asinhf"},
    {"atan2", "atan2"},
    {"atan2f", "atan2f"},
    {"atan", "atan"},
    {"atanf", "atahnf"},
    {"atanh", "atanh"},
    {"atanhf", "atanhf"},
    {"cbrt", "cbrt"},
    {"cbrtf", "cbrtf"},
    {"cos", "cos"},
    {"cosf", "cosf"},
    {"cosh", "cosh"},
    {"coshf", "coshf"},
    {"erfc", "erfc"},
    {"erfcf", "erfcf"},
    {"erf", "erf"},
    {"erff", "erff"},
    {"exp2", "exp2"},
    {"exp2f", "exp2f"},
    {"exp", "exp"},
    {"expf", "expf"},
    {"expm1", "expm1"},
    {"expm1f", "expm1f"},
    {"fmodf", "fmodf"},
    {"fmod", "fmod"},
    {"hypotf", "hypotf"},
    {"hypot", "hypot"},
    {"lgammaf", "lgammaf"},
    {"lgamma", "lgamma"},
    {"llvm.acos.f32", "acosf"},
    {"llvm.acos.f64", "acos"},
    {"llvm.asin.f32", "asinf"},
    {"llvm.asin.f64", "asin"},
    {"llvm.atan.f32", "atanf"},
    {"llvm.atan.f64", "atan"},
    {"llvm.cos.f32", "cosf"},
    {"llvm.cos.f64", "cos"},
    {"llvm.exp.f32", "expf"},
    {"llvm.exp.f64", "exp"},
    {"llvm.fabs.f32", "fabsf"},
    {"llvm.fabs.f64", "fabs"},
    {"llvm.fmod.f32", "fmodf"},
    {"llvm.fmod.f64", "fmod"},
    {"llvm.maxnum.f32", "fmaxf"}, // TODO: Check if this is correct?
    {"llvm.maxnum.f64", "fmax"},  // TODO: Check if this is correct?
    {"llvm.minnum.f32", "fminf"}, // TODO: Check if this is correct?
    {"llvm.minnum.f64", "fmin"},  // TODO: Check if this is correct?
    {"llvm.pow.f32", "powf"},
    {"llvm.pow.f64", "pow"},
    {"llvm.sincos.f32", "sincosf"},
    {"llvm.sincos.f64", "sincos"},
    {"llvm.sin.f32", "sinf"},
    {"llvm.sin.f64", "sin"},
    {"llvm.sqrt.f32", "sqrtf"},
    {"llvm.sqrt.f64", "sqrt"},
    {"llvm.tan.f32", "tanf"},
    {"llvm.tan.f64", "tan"},
    {"llvm.tanh.f32", "tanhf "},
    {"llvm.tanh.f64", "tanh"},
    {"log10f", "log10f"},
    {"log10", "log10"},
    {"log1pf", "log1pf"},
    {"log1p", "log1p"},
    {"log2f", "log2f"},
    {"log2", "log2"},
    {"logf", "logf"},
    {"log", "log"},
    {"powf", "powf"},
    {"pow", "pow"},
    {"sincosf", "sincosf"},
    {"sincos", "sincos"},
    {"sinf", "sinf"},
    {"sinhf", "sinhf"},
    {"sinh", "sinh"},
    {"sin", "sin"},
    {"sqrtf", "sqrtf"},
    {"sqrt", "sqrt"},
    {"tanf", "tanf"},
    {"tanhf", "tanhf"},
    {"tanh", "tanh"},
    {"tan", "tan"},
    {"tgammaf", "tgammaf"},
    {"tgamma", "tgamma"},
};

/// Resolve device functions for hip.
class ResolveDeviceFuncsHip : public ResolveDeviceFuncs {
private:
  static const StringMap<StringRef> devFuncs;

protected:
  std::string getDeviceFunc(StringRef f, bool fast) override final {
    if (devFuncs.find(f) == devFuncs.end())
      return "";

    if (fast)
      llvm_unreachable("NOT YET IMPLEMENTED: Support for fast hip functions");

    // TODO: When fast functions are supported, this will have to change.
    StringRef pfx = fast ? "" : "__ocml_";
    return join_items("", pfx, devFuncs.at(f));
  }

public:
  ResolveDeviceFuncsHip(const TapirTargetOptions &tto)
      : ResolveDeviceFuncs(TTID::Hip, tto) {}
};

const StringMap<StringRef> ResolveDeviceFuncsHip::devFuncs = {
    {"acos", "acos_f64"},
    {"acosf", "acos_f32"},
    {"acosh", "acosh_f64"},
    {"acoshf", "acosh_f32"},
    {"asin", "asin_f64"},
    {"asinf", "asin_f32"},
    {"asinh", "asinh_f64"},
    {"asinhf", "asinh_f32"},
    {"atan2", "atan2_f64"},
    {"atan2f", "atan2_f32"},
    {"atan", "atan_f64"},
    {"atanf", "atahn_f32"},
    {"atanh", "atanh_f64"},
    {"atanhf", "atanh_f32"},
    {"cbrt", "cbrt_f64"},
    {"cbrtf", "cbrt_f32"},
    {"cos", "cos_f64"},
    {"cosf", "cos_f32"},
    {"cosh", "cosh_f64"},
    {"coshf", "cosh_f32"},
    {"erfc", "erfc_f64"},
    {"erfcf", "erfc_f32"},
    {"erf", "erf_f64"},
    {"erff", "erf_f32"},
    {"exp2", "exp2_f64"},
    {"exp2f", "exp2_f32"},
    {"exp", "exp_f64"},
    {"expf", "exp_f32"},
    {"expm1", "expm1_f64"},
    {"expm1f", "expm1_f32"},
    {"fmodf", "fmod_f32"},
    {"fmod", "fmod_f64"},
    {"hypotf", "hypot_f32"},
    {"hypot", "hypot_f64"},
    {"j0f", "j0_f32"},
    {"j0", "j0_f64"},
    {"j1f", "j1_f32"},
    {"j1", "j1_f64"},
    {"lgammaf", "lgamma_f32"},
    {"lgamma", "lgamma_f64"},
    {"llvm.acos.f32", "acos_f32"},
    {"llvm.acos.f64", "acos_f64"},
    {"llvm.asin.f32", "asin_f32"},
    {"llvm.asin.f64", "asin_f64"},
    {"llvm.atan.f32", "atan_f32"},
    {"llvm.atan.f64", "atan_f64"},
    {"llvm.cos.f32", "cos_f32"},
    {"llvm.cos.f64", "cos_f64"},
    {"llvm.exp.f32", "exp_f32"},
    {"llvm.exp.f64", "exp_f64"},
    {"llvm.fabs.f32", "fabs_f32"},
    {"llvm.fabs.f64", "fabs_f64"},
    {"llvm.fmod.f32", "fmod_f32"},
    {"llvm.fmod.f64", "fmod_f64"},
    {"llvm.maxnum.f32", "fmax_f32"}, // TODO: Check if this is correct?
    {"llvm.maxnum.f64", "fmax_f64"}, // TODO: Check if this is correct?
    {"llvm.minnum.f32", "fmin_f32"}, // TODO: Check if this is correct?
    {"llvm.minnum.f64", "fmin_f64"}, // TODO: Check if this is correct?
    {"llvm.pow.f32", "pow_f32"},
    {"llvm.pow.f64", "pow_f64"},
    {"llvm.sincos.f32", "sincos_f32"},
    {"llvm.sincos.f64", "sincos_f64"},
    {"llvm.sin.f32", "sin_f32"},
    {"llvm.sin.f64", "sin_f64"},
    {"llvm.sqrt.f32", "sqrt_f32"},
    {"llvm.sqrt.f64", "sqrt_f64"},
    {"llvm.tan.f32", "tan_f32"},
    {"llvm.tan.f64", "tan_f64"},
    {"llvm.tanh.f32", "tanh_f32 "},
    {"llvm.tanh.f64", "tanh_f64"},
    {"log10f", "log10_f32"},
    {"log10", "log10_f64"},
    {"log1pf", "log1p_f32"},
    {"log1p", "log1p_f64"},
    {"log2f", "log2_f32"},
    {"log2", "log2_f64"},
    {"logf", "log_f32"},
    {"log", "log_f64"},
    {"powf", "pow_f32"},
    {"pow", "pow_f64"},
    {"sincosf", "sincos_f32"},
    {"sincos", "sincos_f64"},
    {"sinf", "sin_f32"},
    {"sinhf", "sinh_f32"},
    {"sinh", "sinh_f64"},
    {"sin", "sin_f64"},
    {"sqrtf", "sqrt_f32"},
    {"sqrt", "sqrt_f64"},
    {"tanf", "tan_f32"},
    {"tanhf", "tanh_f32"},
    {"tanh", "tanh_f64"},
    {"tan", "tan_f64"},
    {"tgammaf", "tgamma_f32"},
    {"tgamma", "tgamma_f64"},
};

} // namespace

namespace llvm {

bool ResolveDeviceFuncsPass::run(TTID tt, Module &devM, Module &hostM,
                                 ModuleAnalysisManager &hostMAM) {
  const TapirTargetInfo &tgi = hostMAM.getResult<TapirTargetAnalysis>(hostM);
  const TapirTargetOptions &tto = tgi.getOptions();

  switch (tt) {
  case TTID::Cuda:
    return ResolveDeviceFuncsCuda(tto).run(devM);
  case TTID::Hip:
    return ResolveDeviceFuncsHip(tto).run(devM);
  default:
    llvm_unreachable("ResolveDeviceFuncsPass: TTID not handled");
  }
}

} // namespace llvm
