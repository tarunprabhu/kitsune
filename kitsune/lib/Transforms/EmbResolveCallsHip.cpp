//===- EmbResolveCallsHip.cpp - Resolve calls to hip libdevice functions --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Resolve calls to library functions for which implementations exist in cuda's
// libdevice module.
//
//===----------------------------------------------------------------------===//

#include "EmbResolveCallsImpl.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/TTUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/AMDGPUEmitPrintf.h"

using namespace llvm;

/// TODO: device-side calls to cover feature set and double-precision support
/// TODO: math options for:
///         - DAZ [on|off],
///         - unsafe math [on|off],
///         - sqrt rounding [on|off],
///         - etc.
///

static const StringMap<StringRef> devFuncs = {
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

static std::string getDeviceFunc(StringRef f, bool fast) {
  if (devFuncs.find(f) == devFuncs.end())
    return "";

  if (fast)
    llvm_unreachable("NOT YET IMPLEMENTED: Support for fast hip functions");

  // TODO: When fast functions are supported, this will have to change.
  StringRef pfx = fast ? "" : "__ocml_";
  return join_items("", pfx, devFuncs.at(f));
}

static Value *emitPrintfCall(CallBase &call, ArrayRef<Value *> args) {
  LLVMContext &ctx = call.getContext();
  IRBuilder irb(ctx);

  BasicBlock *bb = call.getParent()->splitBasicBlockBefore(&call);
  Instruction *term = bb->getTerminator();

  irb.SetInsertPoint(bb);

  // TODO: Do we ever need to use unbuffered output here? Perhaps this should be
  // made configurable in some way - either via the tapir target options, or
  // a hidden command line option.
  Value *result = emitAMDGPUPrintfCall(irb, args, /*IsBuffered=*/true);
  assert(isa<Instruction>(result) && "Result of printf must be an instruction");

  // The last block generated by emitAMDGPUPrintf does not contain a terminator.
  // The terminator of the original basic block into which the printf was
  // inserted must be added to the end of the block containing the result to
  // ensure that the control flow remains correct.
  term->moveAfter(cast<Instruction>(result));

  return result;
}

// Calls to puts have to be handled as a special case. The frontend will have
// stripped one trailing newline from any string literal passed to puts() since
// puts() will always append a newline after emitting the string. When resolving
// this to AMDGPU's printf, we have to ensure that the newline is added back. To
// ensure this, we replace calls to puts as follows:
//
//     puts(s);
//
//  becomes
//
//     printf("%s\n", s);
//
// Returns true if at least one call to puts was replaced, false otherwise.
static bool resolvePutsCalls(Function &f) {
  std::vector<CallBase *> calls;
  for (inst_iterator i = inst_begin(f); i != inst_end(f); ++i)
    if (auto *call = dyn_cast<CallBase>(&*i))
      if (Function *callee = call->getCalledFunction())
        if (callee->getName() == "puts")
          calls.push_back(call);

  if (calls.size()) {
    Module &m = *f.getParent();
    GlobalVariable *fmt = createConstString("%s\n", m);
    for (CallBase *call : calls) {
      std::vector<Value *> args = {fmt};
      for (Value *arg : call->args())
        args.push_back(arg);
      emitPrintfCall(*call, args);
    }

    for (CallBase *call : calls)
      call->eraseFromParent();
  }

  return calls.size();
}

// Calls to printf must be handled as a special case. In principle, we may
// need to do this for other functions that write to stdout as well, but for
// now, we only support printf.
static bool resolvePrintfCalls(Function &f) {
  std::vector<CallBase *> calls;
  for (inst_iterator i = inst_begin(f); i != inst_end(f); ++i)
    if (auto *call = dyn_cast<CallBase>(&*i))
      if (Function *callee = call->getCalledFunction())
        if (callee->getName() == "printf")
          calls.push_back(call);

  for (CallBase *call : calls) {
    std::vector<Value *> args(call->arg_begin(), call->arg_end());
    emitPrintfCall(*call, args);
  }

  for (CallBase *call : calls)
    call->eraseFromParent();

  return calls.size();
}

bool llvm::detail::resolveLibDeviceCallsHip(Module &devM,
                                            const TTOptions &tto) {
  LLVMContext &ctx = devM.getContext();
  std::unique_ptr<Module> libDeviceM = getLibDeviceModule(TTID::Hip, tto, ctx);

  bool changed = false;
  for (Function &f : devM.functions()) {
    if (f.size()) {
      changed |= resolvePutsCalls(f);
      changed |= resolvePrintfCalls(f);
      changed |= resolveCallees(f, *libDeviceM, getDeviceFunc);
    }
  }
  return changed;
}
