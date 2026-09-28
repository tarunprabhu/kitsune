//===- ComputeGPULoopBlockSize.cpp - Iterations per GPU thread ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compute the number of iterations of a tapir loop to execute on each parallel
// thread. Replace calls to the @kit.gpu.ipt intrinsic with this value. We may
// choose to defer this computation to the runtime. In this case, the intrinsic
// will be replaced with a call to a runtime function.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/ComputeGPULoopBlockSize.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Passes/PassUtils.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"

#define DEBUG_TYPE "kit-gpu-loop-block"

using namespace llvm;

static cl::opt<unsigned> clGPULoopBlock("kit-gpu-loop-block", cl::init(1),
                                        cl::desc("GPU loop block size"),
                                        cl::cat(cl::catKitClOpts));

template <typename... Args>
static bool complain(const Loop &loop, DiagID diag, Args &&...args) {
  emitDiagnostic(loop, diag, args...);
  return false;
}

static Value *computeBlockSize(const Loop &root, Type *ty) {
  // FIXME: At this time, we don't yet support choosing an appropriate number of
  // iterations to run per GPU thread. We do allow setting this at compile-time
  // using an LLVM-only argument. Use that if it is provided. Default to a
  // single iteration per thread otherwise.
  if (clGPULoopBlock.getNumOccurrences())
    return ConstantInt::get(ty, clGPULoopBlock, /*isSigned=*/false);
  else
    return ConstantInt::get(ty, 1, /*isSigned=*/false);
}

static void replaceWithConstant(CallInst *call, Value *blockSize) {
  call->replaceAllUsesWith(blockSize);
}

static void replaceWithRuntimeCall(CallInst *call, Value *blockSize) {
  // TODO: Implement this
}

static bool replaceCall(CallInst *call, const Loop &root) {
  Value *blockSize = computeBlockSize(root, call->getType());
  if (isa<ConstantInt>(blockSize))
    replaceWithConstant(call, blockSize);
  else
    replaceWithRuntimeCall(call, blockSize);
  call->eraseFromParent();
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
    return complain(loop, DiagID::ErrLoopNotSimplifyForm);
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
