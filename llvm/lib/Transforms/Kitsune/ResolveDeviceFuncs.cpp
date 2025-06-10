//=- ResolveDeviceFuncsInEmbBC.cpp - Resolve device functions ----*- C++ -*--=//
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

#include "llvm/Transforms/Kitsune/ResolveDeviceFuncs.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/TapirTargetAnalysis.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Kitsune/EmbBCPassUtils.h"
#include "llvm/Transforms/Utils/KitsuneUtils.h"

#define DEBUG_TYPE "resolve-device-funcs"

using namespace llvm;

namespace {

class ResolveDeviceFuncsCuda {
private:
  const TapirTargetOptions &tto;

private:
  StringRef getDeviceFunc(StringRef f) {
    return StringSwitch<StringRef>(f)
        .Case("acos", "acos")
        .Case("acosf", "acosf")
        .Case("acosh", "acosh")
        .Case("acoshf", "acoshf")
        .Case("asin", "asin")
        .Case("asinf", "asinf")
        .Case("asinh", "asinh")
        .Case("asinhf", "asinhf")
        .Case("atan2", "atan2")
        .Case("atan2f", "atan2f")
        .Case("atan", "atan")
        .Case("atanf", "atahnf")
        .Case("atanh", "atanh")
        .Case("atanhf", "atanhf")
        .Case("cbrt", "cbrt")
        .Case("cbrtf", "cbrtf")
        .Case("cos", "cos")
        .Case("cosf", "cosf")
        .Case("cosh", "cosh")
        .Case("coshf", "coshf")
        .Case("erfc", "erfc")
        .Case("erfcf", "erfcf")
        .Case("erf", "erf")
        .Case("erff", "erff")
        .Case("exp2", "exp2")
        .Case("exp2f", "exp2f")
        .Case("exp", "exp")
        .Case("expf", "expf")
        .Case("expm1", "expm1")
        .Case("expm1f", "expm1f")
        .Case("fmodf", "fmodf")
        .Case("fmod", "fmod")
        .Case("hypotf", "hypotf")
        .Case("hypot", "hypot")
        .Case("lgammaf", "lgammaf")
        .Case("lgamma", "lgamma")
        .Case("llvm.cos.f32", "cosf")
        .Case("llvm.cos.f64", "cos")
        .Case("llvm.exp.f32", "expf")
        .Case("llvm.exp.f64", "exp")
        .Case("llvm.fabs.f32", "fabsf")
        .Case("llvm.fabs.f64", "fabs")
        .Case("llvm.fmod.f32", "fmodf")
        .Case("llvm.fmod.f64", "fmod")
        .Case("llvm.maxnum.f32", "fmaxf") // TODO: Check if this is correct?
        .Case("llvm.maxnum.f64", "fmax")  // TODO: Check if this is correct?
        .Case("llvm.minnum.f32", "fminf") // TODO: Check if this is correct?
        .Case("llvm.minnum.f64", "fmin")  // TODO: Check if this is correct?
        .Case("llvm.pow.f32", "powf")
        .Case("llvm.pow.f64", "pow")
        .Case("llvm.sincos.f32", "sincosf")
        .Case("llvm.sincos.f64", "sincos")
        .Case("llvm.sin.f32", "sinf")
        .Case("llvm.sin.f64", "sin")
        .Case("llvm.sqrt.f32", "sqrtf")
        .Case("llvm.sqrt.f64", "sqrt")
        .Case("llvm.tan.f32", "tanf")
        .Case("llvm.tan.f64", "tan")
        .Case("llvm.tanh.f32", "tanhf ")
        .Case("llvm.tanh.f64", "tanh")
        .Case("log10f", "log10f")
        .Case("log10", "log10")
        .Case("log1pf", "log1pf")
        .Case("log1p", "log1p")
        .Case("log2f", "log2f")
        .Case("log2", "log2")
        .Case("logf", "logf")
        .Case("log", "log")
        .Case("powf", "powf")
        .Case("pow", "pow")
        .Case("sincosf", "sincosf")
        .Case("sincos", "sincos")
        .Case("sinf", "sinf")
        .Case("sinhf", "sinhf")
        .Case("sinh", "sinh")
        .Case("sin", "sin")
        .Case("sqrtf", "sqrtf")
        .Case("sqrt", "sqrt")
        .Case("tanf", "tanf")
        .Case("tanhf", "tanhf")
        .Case("tanh", "tanh")
        .Case("tan", "tan")
        .Case("tgammaf", "tgammaf")
        .Case("tgamma", "tgamma")
        .Default("");
  }

  Function *resolveDeviceFunc(Function *f, bool enableFast,
                              Module &libDeviceM) {
    StringRef devFnBase = getDeviceFunc(f->getName());
    if (not devFnBase.empty()) {
      StringRef nvPrefix = enableFast ? "__nv_fast_" : "__nv_";
      std::string devFnName = join_items("", nvPrefix, devFnBase);
      if (Function *devFn = libDeviceM.getFunction(devFnName)) {
        LLVM_DEBUG(dbgs() << "resolve-device-funcs: mapped function '"
                          << f->getName() << "' to '" << devFnName << "'\n");
        // If the device function has already been declared in the kernel
        // module, create a declaration for the device function with the correct
        // attributes.
        Module *m = f->getParent();
        Function *fdecl = m->getFunction(devFnName);
        if (not fdecl) {
          FunctionType *ftype = devFn->getFunctionType();
          GlobalValue::LinkageTypes linkage = devFn->getLinkage();

          fdecl = Function::Create(ftype, linkage, devFnName, m);
          fdecl->setAttributes(devFn->getAttributes());
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
            if (Function *df = resolveDeviceFunc(cf, enableFast, libDeviceM)) {
              call->setCalledFunction(df);
              changed |= true;
            }
          }
        }
      }
    }
    return changed;
  }

public:
  ResolveDeviceFuncsCuda(const TapirTargetOptions &tto) : tto(tto) {}

  bool run(Module &m) {
    std::unique_ptr<Module> libDeviceM =
        parseLibDeviceBCFile(tto.getCudaRuntimeBCFile(), m.getContext());

    bool changed = false;
    for (Function &f : m.functions())
      if (f.size())
        changed |= resolveCallees(f, *libDeviceM);
    return changed;
  }
};

class ResolveDeviceFuncsHip {
private:
  const TapirTargetOptions &tto;

private:
  bool resolveCallees(Function &f) {
    LLVM_DEBUG(dbgs() << "resolve device functions called in " << f.getName());
    bool changed = false;
    // TODO: Implement this.
    return changed;
  }

public:
  ResolveDeviceFuncsHip(const TapirTargetOptions &tto) : tto(tto) {}

  bool run(Module &m) {
    bool changed = false;
    for (Function &f : m.functions())
      if (f.size())
        changed |= resolveCallees(f);
    return changed;
  }
};

} // namespace

namespace llvm {

bool ResolveDeviceFuncsPass::run(TapirTargetID tt, Module &m, Module &hostM,
                                 ModuleAnalysisManager &hostMAM) {
  const TapirTargetInfo &tgi = hostMAM.getResult<TapirTargetAnalysis>(hostM);
  const TapirTargetOptions &tto = tgi.getOptions();
  switch (tt) {
  case TapirTargetID::Cuda:
    return ResolveDeviceFuncsCuda(tto).run(m);
  case TapirTargetID::Hip:
    return ResolveDeviceFuncsHip(tto).run(m);
  default:
    llvm_unreachable("ResolveDeviceFuncsPass: TapirTargetID not handled");
  }
}

} // namespace llvm
