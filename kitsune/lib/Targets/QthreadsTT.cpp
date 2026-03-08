//===- QthreadsTT.cpp - Implementation of the qthreads tapir target -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering to convert Tapir instructions into calls to
// Kitsune's qthreads (POSIX threads) runtime.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Targets/QthreadsTT.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/FunctionUtils.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "qthreadstt"

/// Get the actual grainsize that is to be used. In this tapir target, we do
/// not use a grain size, so always return 0. Otherwise, this will have to be
/// a call to a function from the runtime that calculates the grainsize, or
/// the results of the analysis on the loop that determines an appropriate
/// grainsize to use.
static Value *getGrainSize(Type *type) { return ConstantInt::get(type, 0); }

/// \ingroup kitsune
class QthreadsLoop : public LoopOutlineProcessor {
protected:
  /// Create a wrapper function that has the signature expected by the
  /// qthread_loop function that will run the tapir loop. This does not expect
  /// a grainsize, and calls the function outlined from the tapir loop.
  Function *createWrapperFunctionWithoutGrainsize(Function &outlined) {
    // The arguments of the outlined function are the following:
    //
    //   - out param (optional)
    //   - start
    //   - stop
    //   - grainsize
    //
    // start and stop are the lower and upper bounds of the induction variable
    // of the tapir loop.
    unsigned gsArgNo =
        outlined.hasParamAttribute(0, Attribute::StructRet) ? 3 : 2;

    LLVMContext &ctx = outlined.getContext();
    Type *ret = outlined.getReturnType();
    SmallVector<Type *, 4> params;
    for (Argument &arg : outlined.args())
      if (arg.getArgNo() != gsArgNo)
        params.push_back(arg.getType());
    FunctionType *fty = FunctionType::get(ret, params, /*isVarArg=*/false);

    Module *m = outlined.getParent();
    Twine name = outlined.getName() + ".qthreads.wrapper";
    Function *wrapper = Function::Create(fty, outlined.getLinkage(), name, m);

    copyAttrs(*wrapper, outlined);
    unsigned argNo = 0;
    for (Argument &origArg : outlined.args()) {
      if (origArg.getArgNo() != gsArgNo) {
        copyAttrs(*wrapper->getArg(argNo), origArg);
        ++argNo;
      }
    }

    BasicBlock *bbEntry = BasicBlock::Create(ctx, "entry", wrapper);
    IRBuilder builder(bbEntry);
    SmallVector<Value *, 4> callArgs;
    for (Argument &arg : wrapper->args()) {
      if (arg.getArgNo() == gsArgNo) {
        Type *gsType = outlined.getArg(gsArgNo)->getType();
        Value *gs = getGrainSize(gsType);
        callArgs.push_back(gs);
      }
      callArgs.push_back(&arg);
    }

    CallInst *call = builder.CreateCall(&outlined, callArgs);
    call->setCallingConv(outlined.getCallingConv());
    if (call->getType()->isVoidTy())
      builder.CreateRetVoid();
    else
      builder.CreateRet(call);

    return wrapper;
  }

public:
  /// Create a loop outline processor for the qthreads tapir target.
  /// \param m The host module
  /// \param ttOpts The tapir target options
  QthreadsLoop(Module &m, const TTOptions &ttOpts)
      : LoopOutlineProcessor(m, m, ttOpts,
                             CloneFunctionChangeType::GlobalChanges) {}
  virtual ~QthreadsLoop() = default;

  /// Returns an ArgStructMode enum value describing how inputs to the
  /// underlying task of a tapir loop should be passed to the task.
  ArgStructMode getArgStructMode() const override final {
    // TODO: We should look at the total size of the inputs to the helper
    // function and use a dynamic struct if it is "large".
    return QthreadsTT::ArgStructMode::Static;
  }

  /// Processes a call to an outlined helper function for a tapir loop \p tl.
  void processOutlinedLoopCall(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                               DominatorTree &dt) override final {
    LLVMContext &ctx = M.getContext();

    Constant *ctt = toConstant(TTID::Qthreads, ctx);
    Function *outlined = toi.Outline;
    CallBase *replCall = cast<CallBase>(toi.ReplCall);
    IRBuilder<> builder(replCall);

    Function *wrapper = createWrapperFunctionWithoutGrainsize(*outlined);
    SmallVector<Value *, 16> launchArgs = {ctt, wrapper};
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

QthreadsTT::QthreadsTT(Module &m, const TTOptions &ttOpts)
    : TapirTarget(m, ttOpts) {}

bool QthreadsTT::shouldDoOutlining(const Function &f) const { return true; }

Value *QthreadsTT::lowerGrainsizeCall(CallInst *call) {
  Value *gs = getGrainSize(call->getType());
  call->replaceAllUsesWith(gs);
  return gs;
}

void QthreadsTT::lowerSync(SyncInst &si) {
  // This is only called from the TapirToTarget pass. In some cases, the sync
  // instruction is removed by SimplifyCFG, in which case this is never called.
  // Because of this behavior, we generate a call to __kitqthr_sync()
  // immediately after the call to __kitqthr_launch(). If we do get here, we
  // only need to replace the sync instruction with a simple branch.

  ReplaceInstWithInst(&si, BranchInst::Create(si.getSuccessor(0)));
}

LoopOutlineProcessor *
QthreadsTT::getLoopOutlineProcessor(const TapirLoopInfo *tl) {
  return new QthreadsLoop(M, this->getOptions());
}
