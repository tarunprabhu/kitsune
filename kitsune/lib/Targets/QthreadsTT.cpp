//===- QthreadsTT.cpp - Tapir target that lowers to Qthreads --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tapir target that lowers to Qthreads.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Targets/QthreadsTT.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/FuncUtils.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "qthreadstt"

using namespace llvm;

namespace {

/// \ingroup kitsune
class QthreadsLoop : public LoopOutlineProcessor {
public:
  QthreadsLoop(Module &m, const TTOptions &tto)
      : LoopOutlineProcessor(m, m, tto,
                             CloneFunctionChangeType::GlobalChanges) {}
  virtual ~QthreadsLoop() = default;

  /// Returns an ArgStructMode enum value describing how inputs to the
  /// underlying task of a tapir loop should be passed to the task.
  ArgStructMode getArgStructMode() const override final {
    // TODO: We should look at the total size of the inputs to the helper
    // function and use a dynamic struct if it is "large".
    return QthreadsTT::ArgStructMode::Static;
  }

  /// Setup the loop-control arguments \p lcArgs and loop-control inputs
  /// \p lcInputs for the Tapir loop \p tl.
  void setupLoopControlArgs(TapirLoopInfo *tl, SmallVectorImpl<Value *> &lcArgs,
                            SmallVectorImpl<Value *> &lcInputs) override final {
    assert(tl->getInductionVars()->size() == 1 &&
           "Tapir loop must have exactly one induction variable");

    auto &[iv, ivDescr] = tl->getPrimaryInduction();
    LoopCtlArgs.push_back(new Argument(iv->getType(), "beg"));
    lcArgs.push_back(LoopCtlArgs.back());
    lcInputs.push_back(ivDescr.getStartValue());

    Value *tc = tl->getTripCount();
    assert(tc && "No trip count found for Tapir loop end argument.");
    LoopCtlArgs.push_back(new Argument(tc->getType(), "end"));
    lcArgs.push_back(LoopCtlArgs.back());
    lcInputs.push_back(tc);
  }

  /// Processes a call to an outlined helper function for a tapir loop \p tl.
  void processOutlinedLoopCall(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                               DominatorTree &dt) override final {
    LLVMContext &ctx = M.getContext();

    Constant *ctt = toConstant(TTID::Qthreads, ctx);
    Function *outlined = toi.Outline;
    CallBase *replCall = cast<CallBase>(toi.ReplCall);
    IRBuilder<> builder(replCall);

    // The outlined function does not have a grainsize argument since the
    // function will be passed to qthreads' launch function which does not
    // expect this argument. However, Kitsune's launch_threads intrinsic
    // requires a grainsize. We therefore construct the arguments to the
    // intrinsic manually.
    assert(replCall->getType() == Type::getVoidTy(ctx) &&
           "The outlined function must not return a value");
    assert(replCall->arg_size() == 3 &&
           "Expect outlined function to have exactly 3 arguments");
    Value *start = replCall->getArgOperand(0);
    Value *end = replCall->getArgOperand(1);
    Value *gs = ConstantInt::get(start->getType(), 0);
    Value *args = replCall->getArgOperand(2);
    Value *launchArgs[] = {ctt, outlined, start, end, gs, args};
    (void)builder.CreateIntrinsic(Intrinsic::kit_cpu_threads_launch,
                                  launchArgs);

    assert(replCall->getNumUses() == 0 &&
           "The outlined function must not have any uses");
    replCall->eraseFromParent();
  }
};

} // namespace

QthreadsTT::QthreadsTT(Module &m, const TTOptions &ttOpts)
    : TapirTarget(m, ttOpts) {}

bool QthreadsTT::shouldDoOutlining(const Function &f) const { return true; }

Value *QthreadsTT::lowerGrainsizeCall(CallInst *call) {
  // In this tapir target, we do not use a grain size, so always return 0.
  // Otherwise, this will have to be a call to a function from the runtime that
  // calculates the grainsize, or the results of the analysis on the loop that
  // determines an appropriate grainsize value to use.
  Value *gs = ConstantInt::get(call->getType(), 0);
  call->replaceAllUsesWith(gs);
  return gs;
}

void QthreadsTT::lowerSync(SyncInst &si) {
  // This is only called from the TapirToTarget pass. However, after loop
  // spawning, there will be nothing for that pass to do, so this is not
  // expected to be called. In case it is, fail catastrophically since it would
  // imply that something elsewhere has changed and this may have to be modified
  // to keep up.
  llvm_unreachable("QthreadsTT: Unexpected invocation of lowerSync() callback");
}

LoopOutlineProcessor *
QthreadsTT::getLoopOutlineProcessor(const TapirLoopInfo *tl) {
  return new QthreadsLoop(M, this->getOptions());
}
