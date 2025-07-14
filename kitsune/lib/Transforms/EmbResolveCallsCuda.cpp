//==- EmbResolveCallsCuda.cpp - Resolve calls to cuda libdevice functions --==//
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
#include "kitsune/Transforms/Utils/EmbModulePassUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Module.h"

using namespace llvm;

static const StringMap<StringRef> devFuncs = {
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

static std::string getDeviceFunc(StringRef f, bool fast) {
  if (devFuncs.find(f) == devFuncs.end())
    return "";

  StringRef pfx = fast ? "__nv_fast_" : "__nv_";
  return join_items("", pfx, devFuncs.at(f));
}

bool llvm::detail::resolveLibDeviceCallsCuda(Module &devM,
                                             const TapirTargetOptions &tto) {
  LLVMContext &ctx = devM.getContext();
  std::unique_ptr<Module> libDeviceM = getLibDeviceModule(TTID::Cuda, tto, ctx);

  bool changed = false;
  for (Function &f : devM.functions()) {
    if (f.size()) {
      changed |= resolveCallees(f, *libDeviceM, getDeviceFunc);
    }
  }
  return changed;
}
