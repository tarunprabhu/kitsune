//===- NormalizeLoopControlBlocks.cpp - Normalize loops pre-lowering ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Normalize the control blocks of tapir loops.
//
// This pass ensures that the loop header does not contain any instructions
// other than the PHI node and the terminating detach. Any instructions that
// are safe to sink into the loop body are sunk.
//
// In principle, the same should be done for instructions in the loop latch that
// are not directly related to computing the next value of the canonical loop
// induction variable. However, this has not yet been implemented because it is
// not clear if it is safe to raise such code into the loop body.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/NormalizeLoopControlBlocks.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/IR/Instructions.h"

#define DEBUG_TYPE "kit-loop-ctrl"

using namespace llvm;

// The header of a tapir loop must contain only PHI nodes. Try to sink any
// other instructions present in the header into the body. The instructions will
// be sunk only if all non-PHI instructions in the header are safe to sink to
// the body. In other words, either all non-PHI instructions will be sunk, or
// none will be.
static bool sinkHeaderInsts(Loop &loop) {
  auto isSafeToSinkIntoBody = [](const Instruction *inst) -> bool {
    // This is a very stupid check. We should do something a bit more
    // sophisticated, but this seems to work for now.
    return isa<BinaryOperator>(inst) || isa<CastInst>(inst) ||
           isa<UnaryOperator>(inst);
  };

  SmallVector<Instruction *, 4> insts;
  for (Instruction &inst : *loop.getHeader())
    if (!isa<PHINode>(inst) && !isa<DetachInst>(inst))
      insts.push_back(&inst);

  if (std::all_of(insts.begin(), insts.end(), isSafeToSinkIntoBody)) {
    LLVM_DEBUG(dbgs() << "prelower-prepare: sinking " << insts.size()
                      << " instructions into body\n");
    BasicBlock *body = getTapirLoopDetachedBlock(loop);
    BasicBlock::iterator insertPt = body->getFirstNonPHIOrDbg();
    for (Instruction *inst : insts) {
      LLVM_DEBUG(dbgs() << "NormalizeLoopControl: Sinking " << *inst << "\n");
      inst->moveBefore(insertPt);
    }
    return true;
  }
  return false;
}

// The latch must only contain instructions that are directly related to
// updating the loop induction variable, or computing the loop termination
// condition. Raise any other instructions into the loop body.
static bool raiseLatchInsts(Loop &loop) {
  // TODO: Implement this.
  return false;
}

PreservedAnalyses
NormalizeLoopControlBlocksPass::run(Loop &loop, LoopAnalysisManager &am,
                                    LoopStandardAnalysisResults &ar,
                                    LPMUpdater &updater) {
  bool changed = false;

  if (isTapirLoop(loop)) {
    LLVM_DEBUG(dbgs() << "NormalizeLoopControl: Found loop '" << getName(loop)
                      << "'\n");
    changed |= sinkHeaderInsts(loop);
    changed |= raiseLatchInsts(loop);
  }

  if (!changed)
    return PreservedAnalyses::all();
  return getLoopPassPreservedAnalyses();
}
