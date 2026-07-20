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
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

CPUTTLoopProcessor::CPUTTLoopProcessor(TTID tt, const TTOptions &opts,
                                       bool asyncLaunch, Module &m)
    : LoopOutlineProcessor(m, m, opts, CloneFunctionChangeType::GlobalChanges),
      tt(tt), asyncLaunch(asyncLaunch) {}

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

Function *CPUTTLoopProcessor::genWrapperFor(Function &outlined) {
  LLVMContext &ctx = outlined.getContext();

  FunctionType *outlinedTy = outlined.getFunctionType();
  Type *begTy = outlinedTy->getParamType(0);
  Type *endTy = outlinedTy->getParamType(1);

  Type *ptrTy = PointerType::getUnqual(ctx);
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *wrapperTy =
      FunctionType::get(voidTy, {begTy, endTy, ptrTy}, /*isVarArg=*/false);

  SmallVector<Type *, 4> bundleTys;
  for (unsigned i = 2; i < outlined.arg_size(); ++i)
    bundleTys.push_back(outlinedTy->getParamType(i));
  StructType *bundleTy = StructType::get(ctx, bundleTys);

  Module &m = *outlined.getParent();
  Twine wrapperName = outlined.getName() + ".wrapper";
  Function *wrapper = cast<Function>(
      m.getOrInsertFunction(wrapperName.str(), wrapperTy).getCallee());
  wrapper->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
  wrapper->setCallingConv(CallingConv::Fast);

  Value *beg = wrapper->getArg(0);
  Value *end = wrapper->getArg(1);
  Value *bundle = wrapper->getArg(2);

  beg->setName("beg");
  end->setName("end");
  bundle->setName("args");

  BasicBlock *bbEntry = BasicBlock::Create(ctx, "entry", wrapper);
  BasicBlock *bbExit = BasicBlock::Create(ctx, "exit", wrapper);
  IRBuilder<> builder(ctx);

  builder.SetInsertPoint(bbEntry);
  SmallVector<Value *, 4> outlinedArgs = {beg, end};
  for (unsigned i = 0; i < bundleTys.size(); ++i) {
    Type *elemTy = bundleTys[i];
    StringRef name = outlined.getArg(i + 2)->getName();
    Value *off = builder.CreateConstInBoundsGEP2_32(bundleTy, bundle, 0, i);
    Value *load = builder.CreateLoad(elemTy, off, name);
    outlinedArgs.push_back(load);
  }
  CallInst *call =
      cast<CallInst>(builder.CreateCall(outlinedTy, &outlined, outlinedArgs));
  call->setCallingConv(outlined.getCallingConv());
  builder.CreateBr(bbExit);

  builder.SetInsertPoint(bbExit);
  builder.CreateRetVoid();

  // Inline the outlined function into the wrapper. We merge the attributes
  // from the outlined function with those of the wrapper since the latter is
  // effectively a replacement for the former.
  InlineFunctionInfo ifi;
  InlineResult result = InlineFunction(*call, ifi, /*MergeAttributes=*/true);
  assert(result.isSuccess() &&
         "Inlining outlined function into wrapper failed");

  return wrapper;
}

void CPUTTLoopProcessor::processOutlinedLoopCall(TapirLoopInfo &tl,
                                                 TaskOutlineInfo &toi,
                                                 DominatorTree &dt) {
  CallBase *replCall = cast<CallBase>(toi.ReplCall);
  assert(replCall->getType()->isVoidTy() &&
         "The outlined function must not return a value");
  assert(replCall->arg_size() >= 2 &&
         "Expect outlined function to have at least two arguments");
  assert(replCall->getCalledFunction() &&
         "Outlined function must be called directly");

  // The first two arguments of the call are the lower and upper bounds of the
  // iteration range. The rest are other entities used in the outlined body of
  // the parallel loop that was outlined. This outlined function will be
  // executed by each thread. The underlying runtimes, such as OpenMP and
  // Qthreads, however, any additional arguments to be bundled into a single
  // struct. We, therefore, generate a wrapper function that will receive such
  // an argument bundle. That original outlined function will be inlined into
  // this wrapper.
  Function *outlined = replCall->getCalledFunction();
  Function *wrapper = genWrapperFor(*outlined);

  Value *beg = replCall->getArgOperand(0);
  Value *end = replCall->getArgOperand(1);
  Constant *ctt = toConstant(tt, replCall->getContext());
  SmallVector<Value *, 4> launchArgs = {ctt, wrapper, beg, end};
  for (unsigned i = 2; i < replCall->arg_size(); ++i)
    launchArgs.push_back(replCall->getArgOperand(i));

  IRBuilder<> builder(replCall);
  if (asyncLaunch) {
    Value *launchCtx = builder.CreateIntrinsic(
        Intrinsic::kit_async_cpu_threads_launch, launchArgs);
    Value *syncArgs[] = {ctt, launchCtx};
    builder.CreateIntrinsic(Intrinsic::kit_cpu_threads_sync, syncArgs);
  } else {
    builder.CreateIntrinsic(Intrinsic::kit_cpu_threads_launch, launchArgs);
  }

  assert(replCall->getNumUses() == 0 &&
         "The outlined function must not have any uses");
  replCall->eraseFromParent();
}
