//===- PthreadsTT.cpp - Interface to Kitsune's pthreads runtime -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering to convert Tapir instructions into calls to
// Kitsune's pthreads (POSIX threads) runtime.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/PthreadsTT.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "pthreadstt"

PthreadsLoop::PthreadsLoop(Module &m, const TapirTargetOptions &ttOpts)
    : LoopOutlineProcessor(m, m, ttOpts,
                           CloneFunctionChangeType::GlobalChanges) {}

PthreadsLoop::~PthreadsLoop() {}

PthreadsTT::ArgStructMode PthreadsLoop::getArgStructMode() const {
  // TODO: We should look at the total size of the inputs to the helper function
  // and use a dynamic struct if it is "large".
  return PthreadsTT::ArgStructMode::Static;
}

void PthreadsLoop::processOutlinedLoopCall(TapirLoopInfo &tl,
                                           TaskOutlineInfo &toi,
                                           DominatorTree &dt) {
  LLVMContext &ctx = M.getContext();

  ConstantInt *ctt = createConstInt(TTID::Pthreads, ctx);
  Function* outlined = toi.Outline;
  CallBase *replCall = cast<CallBase>(toi.ReplCall);
  IRBuilder<> builder(replCall);

  SmallVector<Value*, 16> launchArgs = {ctt, outlined};
  for (Value* arg : replCall->args())
    launchArgs.push_back(arg);
  Value *thrdCtx =
      builder.CreateIntrinsic(Intrinsic::kit_async_launch_threads, launchArgs);

  Value *syncArgs[] = {ctt, thrdCtx};
  (void)builder.CreateIntrinsic(Intrinsic::kit_sync_threads, syncArgs);

  assert(replCall->getType() == Type::getVoidTy(ctx) &&
         "The outlined function must not return a value");
  assert(replCall->getNumUses() == 0 &&
         "The outlined function must not have any uses");
  replCall->eraseFromParent();
}

PthreadsTT::PthreadsTT(Module &m, const TapirTargetOptions &ttOpts)
    : TapirTarget(m, ttOpts) {}

bool PthreadsTT::shouldDoOutlining(const Function &f) const { return true; }

Value *PthreadsTT::lowerGrainsizeCall(CallInst *call) {
  // We don't use a grain size in this tapir target, so set it to zero to
  // indicate that it is unused.
  Value *zero = ConstantInt::get(call->getType(), 0);
  call->replaceAllUsesWith(zero);
  return zero;
}

void PthreadsTT::lowerSync(SyncInst &si) {
  // This is only called from the TapirToTarget pass. In some cases, the sync
  // instruction is removed by SimplifyCFG, in which case this is never called.
  // Because of this behavior, we generate a call to __kitpthr_sync()
  // immediately after the call to __kitpthr_launch(). If we do get here, we
  // only need to replace the sync instruction with a simple branch.

  ReplaceInstWithInst(&si, BranchInst::Create(si.getSuccessor(0)));
}

LoopOutlineProcessor *
PthreadsTT::getLoopOutlineProcessor(const TapirLoopInfo *TL) {
  return new PthreadsLoop(M, this->getOptions());
}
