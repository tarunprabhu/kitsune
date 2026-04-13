//===- OpenMPTT.cpp - Tapir target that lowers to LLVM's OpenMP runtime ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tapir target that lowers to LLVM's OpenMP runtime.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Targets/OpenMPTT.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "openmptt"

using namespace llvm;

namespace {

/// \ingroup kitsune
class OpenMPLoop : public LoopOutlineProcessor {
public:
  /// Create a loop outline processor for the openmp tapir target.
  /// \param m The host module
  /// \param ttOpts The tapir target options
  OpenMPLoop(Module &m, const TTOptions &tto)
      : LoopOutlineProcessor(m, m, tto,
                             CloneFunctionChangeType::GlobalChanges) {}
  virtual ~OpenMPLoop() = default;

  /// Returns an ArgStructMode enum value describing how inputs to the
  /// underlying task of a tapir loop should be passed to the task.
  ArgStructMode getArgStructMode() const override final {
    // TODO: We should look at the total size of the inputs to the helper
    // function and use a dynamic struct if it is "large".
    return OpenMPTT::ArgStructMode::Static;
  }

  /// Processes a call to an outlined helper function for a tapir loop \p tl.
  void processOutlinedLoopCall(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                               DominatorTree &dt) override final {
    LLVMContext &ctx = M.getContext();

    Constant *ctt = toConstant(TTID::OpenMP, ctx);
    Function *outlined = toi.Outline;
    CallBase *replCall = cast<CallBase>(toi.ReplCall);
    IRBuilder<> builder(replCall);

    SmallVector<Value *, 16> launchArgs = {ctt, outlined};
    for (Value *arg : replCall->args())
      launchArgs.push_back(arg);
    (void)builder.CreateIntrinsic(Intrinsic::kit_launch_threads, launchArgs);

    assert(replCall->getType() == Type::getVoidTy(ctx) &&
           "The outlined function must not return a value");
    assert(replCall->getNumUses() == 0 &&
           "The outlined function must not have any uses");
    replCall->eraseFromParent();
  }
};

} // namespace

OpenMPTT::OpenMPTT(Module &m, const TTOptions &tto) : TapirTarget(m, tto) {}

bool OpenMPTT::shouldDoOutlining(const Function &f) const { return true; }

Value *OpenMPTT::lowerGrainsizeCall(CallInst *call) {
  /// In this tapir target, we do not use a grain size, so always return 0.
  /// Otherwise, this will have to be a call to a function from the runtime that
  /// calculates the grainsize, or the results of the analysis on the loop that
  /// determines an appropriate grainsize value to use.
  Constant *gs = ConstantInt::get(call->getType(), 0);
  call->replaceAllUsesWith(gs);
  return gs;
}

void OpenMPTT::lowerSync(SyncInst &si) {
  // This is only called from the TapirToTarget pass. However, after loop
  // spawning, there will be nothing for that pass to do, so this is not
  // expected to be called. In case it is, fail catastrophically since it would
  // imply that something elsewhere has changed and this may have to be modified
  // to keep up.
  llvm_unreachable("OpenMPTT: Unexpected invocation of lowerSync() callback");
}

LoopOutlineProcessor *
OpenMPTT::getLoopOutlineProcessor(const TapirLoopInfo *tl) {
  return new OpenMPLoop(M, this->getOptions());
}
