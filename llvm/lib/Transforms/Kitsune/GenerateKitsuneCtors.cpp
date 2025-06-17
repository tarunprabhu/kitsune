//=- GenerateKitsuneCtor.cpp - Generate global ctors for Kitsune --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of pass that generates global constructors for Kitsune's
// runtime
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Kitsune/GenerateKitsuneCtors.h"
#include "GenerateKitsuneCtorsImpl.h"
#include "llvm/Analysis/TapirTargetAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

using namespace llvm;

/// Should a ctor be generated for a GPU-centric tapir target. To determine if
/// this is the case, check that at least one call to Kitsune's launch kernel
/// intrinsic is present in the module.
static bool shouldGenerateGPUCtor(Module &m, TapirTargetID tt) {
  assert((tt == TapirTargetID::Cuda || tt == TapirTargetID::Hip) &&
         "shouldGenerateGPUCtor: Tapir target must be GPU-centric");
  StringRef launch = Intrinsic::getBaseName(Intrinsic::kitrt_launch_kernel);
  if (Function *f = m.getFunction(launch)) {
    for (Use &u : f->uses()) {
      if (auto *call = dyn_cast<CallBase>(u.getUser())) {
        // Although unlikely, the intrinsic could have been passed as an
        // argument to some other function. Just in case, check that the callee
        // at this site is the launch kernel function.
        if (call->getIntrinsicID() == Intrinsic::kitrt_launch_kernel) {
          auto *cint = dyn_cast<ConstantInt>(call->getArgOperand(0));
          if (cint->getZExtValue() == unsigned(tt))
            return true;
        }
      }
    }
  }
  return false;
}

namespace llvm {

PreservedAnalyses GenerateKitsuneCtorsPass::run(Module &m,
                                                ModuleAnalysisManager &mam) {
  // If no primary tapir target has been set, there will be nothing to do, so
  // bail out immediately.
  const TapirTargetInfo &tgi = mam.getResult<TapirTargetAnalysis>(m);
  if (not tgi.hasID())
    return PreservedAnalyses::all();

  auto &fam = mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();
  auto getTLI = [&](Function &f) -> TargetLibraryInfo & {
    return fam.getResult<TargetLibraryAnalysis>(f);
  };
  const TapirTargetOptions &tto = tgi.getOptions();

  if (shouldGenerateGPUCtor(m, TapirTargetID::Cuda))
    detail::genKitsuneCtorCuda(m, tto, getTLI);

  if (shouldGenerateGPUCtor(m, TapirTargetID::Hip))
    detail::genKitsuneCtorHip(m, tto, getTLI);

  // This never invalidates any analyses since only a global variable will have
  // changed. The generated ctors will not be called explicitly in the code,
  // so the callgraph will not have changed either.
  return PreservedAnalyses::all();
}

} // namespace llvm
