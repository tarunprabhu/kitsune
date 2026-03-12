//===- SerialTT.cpp - Implementation of the serial tapir target -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tapir target that serializes tapir loops.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Targets/SerialTT.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "serialtt"

namespace {

/// \ingroup kitsune
class SerialLoop : public LoopOutlineProcessor {
public:
  SerialLoop(Module &m, const TTOptions &opts)
      : LoopOutlineProcessor(m, m, opts,
                             CloneFunctionChangeType::GlobalChanges) {}
  virtual ~SerialLoop() = default;

  virtual void preProcessTapirLoop(TapirLoopInfo &tl,
                                   ValueToValueMapTy &vmap) override final {
    Task *task = tl.getTask();
    DetachInst *detach = task->getDetach();
    SerializeDetach(detach, task);
  }
};

} // namespace

SerialTT::SerialTT(Module &m, const TTOptions &ttOpts)
    : TapirTarget(m, ttOpts) {}

bool SerialTT::shouldDoOutlining(const Function &f) const { return false; }

Value *SerialTT::lowerGrainsizeCall(CallInst *call) {
  /// Get the actual grainsize that is to be used. In this tapir target, we do
  /// not use a grain size, so always return 0.
  Value *gs = ConstantInt::get(call->getType(), 0);
  call->replaceAllUsesWith(gs);
  return gs;
}

void SerialTT::lowerSync(SyncInst &si) {
  // This is only called from the TapirToTarget pass. In some cases, the sync
  // instruction is removed by SimplifyCFG, in which case this is never called.
  ReplaceInstWithInst(&si, BranchInst::Create(si.getSuccessor(0)));
}

LoopOutlineProcessor *
SerialTT::getLoopOutlineProcessor(const TapirLoopInfo *tl) {
  return new SerialLoop(M, this->getOptions());
}
