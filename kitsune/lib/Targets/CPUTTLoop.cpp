//===- CPUTTLoop.cpp - CPU-centric loop outline processor -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for loop outline processors used by the CPU-centric
// threading-focused tapir target such as openmp, pthreads, and qthreads.
//
//===----------------------------------------------------------------------===//

#include "CPUTTLoop.h"
#include "kitsune/Core/ConstantUtils.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

using namespace llvm;

CPUTTLoopProcessor::CPUTTLoopProcessor(TTID tt, const TTOptions &opts,
                                       bool asyncLaunch, Module &m)
    : LoopOutlineProcessor(m, m, opts, CloneFunctionChangeType::GlobalChanges),
      tt(tt), asyncLaunch(asyncLaunch) {}

CPUTTLoopProcessor::ArgStructMode CPUTTLoopProcessor::getArgStructMode() const {
  // TODO: We should look at the total size of the inputs to the helper
  // function and use a dynamic struct if it is "large".
  return CPUTTLoopProcessor::ArgStructMode::Static;
}

void CPUTTLoopProcessor::setupLoopControlArgs(
    TapirLoopInfo *tl, SmallVectorImpl<Value *> &lcArgs,
    SmallVectorImpl<Value *> &lcInputs) {
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

void CPUTTLoopProcessor::insertAsyncLaunch(CallBase &call) {
  LLVMContext &ctx = call.getContext();
  Constant *ctt = toConstant(tt, ctx);
  Value *outlined = call.getCalledOperand();
  IRBuilder<> builder(&call);

  SmallVector<Value *, 4> launchArgs = {ctt, outlined};
  for (Value *arg : call.args())
    launchArgs.push_back(arg);

  Value *launchCtx = builder.CreateIntrinsic(
      Intrinsic::kit_async_cpu_threads_launch, launchArgs);

  Value *syncArgs[] = {ctt, launchCtx};
  (void)builder.CreateIntrinsic(Intrinsic::kit_cpu_threads_sync, syncArgs);
}

void CPUTTLoopProcessor::insertBlockingLaunch(CallBase &call) {
  LLVMContext &ctx = call.getContext();
  Constant *ctt = toConstant(tt, ctx);
  Value *outlined = call.getCalledOperand();
  IRBuilder<> builder(&call);

  SmallVector<Value *, 4> launchArgs = {ctt, outlined};
  for (Value *arg : call.args())
    launchArgs.push_back(arg);

  (void)builder.CreateIntrinsic(Intrinsic::kit_cpu_threads_launch, launchArgs);
}

void CPUTTLoopProcessor::processOutlinedLoopCall(TapirLoopInfo &tl,
                                                 TaskOutlineInfo &toi,
                                                 DominatorTree &dt) {
  CallBase *replCall = cast<CallBase>(toi.ReplCall);
  assert(replCall->getType()->isVoidTy() &&
         "The outlined function must not return a value");
  assert(replCall->arg_size() == 3 &&
         "Expect outlined function to have exactly 3 arguments");

  if (asyncLaunch)
    insertAsyncLaunch(*replCall);
  else
    insertBlockingLaunch(*replCall);

  assert(replCall->getNumUses() == 0 &&
         "The outlined function must not have any uses");
  replCall->eraseFromParent();
}
