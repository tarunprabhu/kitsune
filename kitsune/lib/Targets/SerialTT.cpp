//===- SerialTT.cpp - Tapir target that serializes tapir loops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tapir target that serializes tapir loops.
//
// Since the serial tapir target explicitly disables loop outlining, the tapir
// loops are simply ignored by the loop spawning pass that drives the tapir
// targets. That leaves us with only two reasonable options: the
// preProcessFunction and postProcessFunction callbacks.
//
// If the tapir loops were in preProcessFunction, the loop spawning pass would
// have to be modified to drop the serialized loops from its internal data
// structures. It's not bad, but given that we are being forced to compromise
// anyway, it is, arguably, a bit too much.
//
// Since the tapir loops will have been ignored by loop-spawning, they will
// still be around when postProcessFunction is called. However, serializing them
// here has other drawbacks. For one, to do so, we need the results of
// TaskAnalysis. Unlike with preProcessFunction, this analysis is *NOT* passed
// to postProcessFunction (this is not unreasonable since this analysis may have
// been invalidated during lowering). This requires us to recompute TaskInfo in
// postProcessFunction before serializing the tapir loops.
//
// Since serializing the loops in postProcessFunction can be done without
// breaking loop-spawning, this is what we have done.
//
// Unlike earlier implementations of this tapir target that nearly always caused
// conflicts with loop spawning when something about the serialization process
// was changed, this may actually be more stable. The major question that
// remains unanswered is what happens when multi-target execution is enabled.
// This decision may well have to be revisited in that case.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Targets/SerialTT.h"
#include "kitsune/Core/IntrinsicUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "serialtt"

SerialTT::SerialTT(Module &m, const TTOptions &tto) : TapirTarget(m, tto) {}

bool SerialTT::shouldDoOutlining(const Function &f) const { return false; }

Value *SerialTT::lowerGrainsizeCall(CallInst *call) {
  // This callback is not called by the loop-spawning pass - it is only called
  // by tapir-to-target. Since this pass serializes all tapir loops, there
  // should be nothing for tapir-to-target to find, so this should never be
  // called.
  //
  // Fail catastrophically if it is called since it likely means that something
  // significant has changed elsewhere in the pipeline.
  llvm_unreachable(
      "Did not expect SerialTT::lowerGrainsizeCall() to be called");
}

void SerialTT::lowerSync(SyncInst &si) {
  // This callback is not called by the loop-spawning pass - it is only called
  // by tapir-to-target. Since this pass serializes all tapir loops, there
  // should be nothing for tapir-to-target to find, so this should never be
  // called.
  //
  // Fail catastrophically if it is called since it likely means that something
  // significant has changed elsewhere in the pipeline.
  llvm_unreachable("Did not expect SerialTT::lowerSync() to be called");
}

void SerialTT::postProcessFunction(Function &f, bool processingTapirLoops) {
  // FIXME: It would be good to pass these analyses into this callback - or use
  // some other mechanism to retrieve these analyses from the loop spawning
  // pass that drives the tapir targets - rather than computing them here.
  TaskInfo ti;
  DominatorTree dt(f);
  LoopInfo li(dt);

  // Lower calls to llvm.kit.cpu.num.threads intrinsic. We want this tapir
  // target to produce code that, to the extent possible, looks as if it was the
  // result of compiling a standard serial loop. Since some simplification
  // passes are run after the loop-spawning pass, lowering this intrinsic here
  // will ensure that that is the case.
  if (Function *numThreadsFn = Intrinsic::getDeclarationIfExists(
          &M, Intrinsic::kit_cpu_num_threads)) {
    std::vector<CallInst *> calls;
    // Only replace calls in this function. Although unlikely, the intrinsic
    // could be passed as an argument to another function, so check that it is
    // actually called where it is used.
    for (Use &u : numThreadsFn->uses())
      if (auto *call = dyn_cast<CallInst>(u.getUser()))
        if (call->getFunction() == &f &&
            call->getIntrinsicID() == Intrinsic::kit_cpu_num_threads &&
            *getTTIDFromKitIntrCall(*call) == TTID::Serial)
          calls.push_back(call);

    Type *type = numThreadsFn->getReturnType();
    Value *one = ConstantInt::get(type, 1, /*isSigned=*/false);
    for (CallInst *call : calls) {
      call->replaceAllUsesWith(one);
      call->eraseFromParent();
    }
  }

  ti.recalculate(f, dt);

  for (Loop *loop : li.getLoopsInPreorder())
    if (getTargetAttr(*loop) == TTID::Serial)
      if (Task *task = getTaskIfTapirLoop(loop, &ti))
        serializeTapirLoop(*loop, *task);
}
