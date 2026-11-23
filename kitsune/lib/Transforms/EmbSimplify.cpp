//===- EmbSimplify.cpp - Simplify functions in embedded modules -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Run a sequence of function simplification passes on functions in embedded
// modules.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EmbSimplify.h"
#include "kitsune/Core/FuncAttrs.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

#define DEBUG_TYPE "emb-simplify"

using namespace llvm;

static cl::opt<bool>
    clSimplifyAll("emb-simplify-all", cl::init(false), cl::Hidden,
                  cl::desc("Run the simplification passes on all functions in "
                           "the embedded module, not just the kernel function"),
                  cl::cat(cl::catKitClDevOpts));

bool EmbSimplifyPass::run(TTID tt, Function &devF,
                          FunctionAnalysisManager &devAM, Module &hostM,
                          ModuleAnalysisManager &hostAM) {
  FunctionPassManager fpm;

  fpm.addPass(InstCombinePass());
  fpm.addPass(SimplifyCFGPass(SimplifyCFGOptions()
                                  .convertSwitchRangeToICmp(true)
                                  .convertSwitchToArithmetic(true)
                                  .hoistCommonInsts(true)
                                  .sinkCommonInsts(true)));
  fpm.addPass(ADCEPass());

  if (clSimplifyAll || hasKernelAttr(devF))
    return !fpm.run(devF, devAM).areAllPreserved();
  return false;
}
