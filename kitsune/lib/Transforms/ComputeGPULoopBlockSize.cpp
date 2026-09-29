//===- ComputeGPULoopBlockSize.cpp - Iterations per GPU thread ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compute the number of iterations of a tapir loop to execute on each parallel
// thread. This will replace the second argument in calls to the `kit.gpu.ipt`
// intrinsic with an appropriate value.
//
// The `kit-prepare` pass may have inserted calls to this intrinsic in a tapir
// loop's preheader. This pass is intended to run immediately after that one, so
// we can be certain that the calls will be in the preheader.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/ComputeGPULoopBlockSize.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Passes/PassUtils.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/GlobalVariable.h"

#define DEBUG_TYPE "kit-gpu-loop-block"

using namespace llvm;

static cl::opt<uint64_t> clGPULoopBlock("kit-gpu-loop-ipt", cl::init(1),
                                        cl::desc("GPU loop block size"),
                                        cl::cat(cl::catKitClOpts));

static Constant *getGlobalConstant(Module &m, Type *ty, uint64_t n) {
  for (GlobalVariable &g : m.globals())
    if (g.hasInitializer())
      if (auto *init = dyn_cast<ConstantInt>(g.getInitializer()))
        if (init->getType() == ty && init->getLimitedValue() == n)
          return &g;

  Constant *init = ConstantInt::get(ty, n, /*isSigned=*/false);
  return new GlobalVariable(m, ty, /*isConstant=*/true,
                            GlobalValue::InternalLinkage, init);
}

static Constant *computeBlockSize(Loop &loop, Type *ty) {
  // FIXME: At this time, we don't yet support choosing an appropriate number of
  // iterations to run per GPU thread. We do allow setting this at compile-time
  // using an LLVM-only argument. Use that if it is provided. Default to a
  // single iteration per thread otherwise.
  Module &m = *getModule(loop);
  if (clGPULoopBlock.getNumOccurrences())
    return getGlobalConstant(m, ty, clGPULoopBlock);
  else
    return getGlobalConstant(m, ty, 1);
}

static bool replaceCall(CallInst *call, Loop &loop) {
  Constant *blockSize = computeBlockSize(loop, call->getType());
  call->setArgOperand(1, blockSize);
  return true;
}

static bool runOnLoop(Loop &loop) {
  BasicBlock &ph = *loop.getLoopPreheader();
  for (Instruction &inst : ph)
    if (CallInst *call = dyn_cast<CallInst>(&inst))
      if (call->getIntrinsicID() == Intrinsic::kit_gpu_ipt)
        return replaceCall(call, loop);
  return false;
}

static bool check(const Loop &loop) {
  if (!loop.isLoopSimplifyForm())
    return emitDiagnostic(loop, DiagID::ErrLoopNotSimplifyForm);
  return true;
}

PreservedAnalyses
ComputeGPULoopBlockSizePass::run(Loop &loop, LoopAnalysisManager &am,
                                 LoopStandardAnalysisResults &ar,
                                 LPMUpdater &updater) {
  if (check(loop))
    runOnLoop(loop);

  // This pass will replace calls to an intrinsic with either a compile-time
  // constant, or a call to a runtime function. In either case, only the
  // callgraph will be affected.
  return getPreservedAnalysesCallGraph();
}
