//===- PthreadsTT.cpp - Tapir target that lowers to pthreads --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tapir target that lowers to POSIX threads (pthreads).
//
//===----------------------------------------------------------------------===//

#include "kitsune/Targets/PthreadsTT.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "pthreadstt"

using namespace llvm;

namespace {

/// \ingroup kitsune
class PthreadsLoop : public LoopOutlineProcessor {
public:
  /// Create a loop outline processor for the pthreads tapir target.
  /// \param m The host module
  /// \param ttOpts The tapir target options
  PthreadsLoop(Module &m, const TTOptions &ttOpts)
      : LoopOutlineProcessor(m, m, ttOpts,
                             CloneFunctionChangeType::GlobalChanges) {}

  ~PthreadsLoop() = default;

  /// Returns an ArgStructMode enum value describing how inputs to the
  /// underlying task of a tapir loop should be passed to the task.
  ArgStructMode getArgStructMode() const override final {
    // TODO: We should look at the total size of the inputs to the helper
    // function and use a dynamic struct if it is "large".
    return PthreadsTT::ArgStructMode::Static;
  }

  /// Processes a call to an outlined helper function for a tapir loop \p tl.
  void processOutlinedLoopCall(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                               DominatorTree &dt) override final {
    LLVMContext &ctx = M.getContext();

    Constant *ctt = toConstant(TTID::Pthreads, ctx);
    Function *outlined = toi.Outline;
    CallBase *replCall = cast<CallBase>(toi.ReplCall);
    IRBuilder<> builder(replCall);

    SmallVector<Value *, 16> launchArgs = {ctt, outlined};
    for (Value *arg : replCall->args())
      launchArgs.push_back(arg);
    Value *thrdCtx = builder.CreateIntrinsic(
        Intrinsic::kit_async_launch_threads, launchArgs);

    Value *syncArgs[] = {ctt, thrdCtx};
    (void)builder.CreateIntrinsic(Intrinsic::kit_sync_threads, syncArgs);

    assert(replCall->getType() == Type::getVoidTy(ctx) &&
           "The outlined function must not return a value");
    assert(replCall->getNumUses() == 0 &&
           "The outlined function must not have any uses");
    replCall->eraseFromParent();
  }
};

} // namespace

PthreadsTT::PthreadsTT(Module &m, const TTOptions &ttOpts)
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
PthreadsTT::getLoopOutlineProcessor(const TapirLoopInfo *tl) {
  return new PthreadsLoop(M, this->getOptions());
}
