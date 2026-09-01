//===- LowerReduceIntrinsics.cpp - Lower Kitsune's reduce intrinsics ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune's reduce intrinsics.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/LowerReduceIntrinsics.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/Reductions.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "kit-lower-reduce-intrinsics"

using namespace llvm;

using LowerFunc = bool(CallInst *);

static const DenseMap<Intrinsic::ID, LowerFunc *> lowerFuncs = {
    // Reduce intrinsics
    {Intrinsic::kit_reduce_0, detail::lowerReduce0Intr},
};

static bool lowerIntrinsics(Module &m) {
  SmallVector<CallInst *, 0> calls;
  for (Function &f : m)
    for (BasicBlock &bb : f)
      for (Instruction &inst : bb)
        if (auto *call = dyn_cast<CallInst>(&inst))
          if (lowerFuncs.contains(call->getIntrinsicID()))
            calls.push_back(call);

  for (CallInst *call : calls)
    if (Intrinsic::ID id = call->getIntrinsicID())
      lowerFuncs.at(id)(call);

  return calls.size();
}

bool llvm::detail::lowerReduce0Intr(CallInst *call) {
  ReductionInfo redxn(call);
  FunctionType *reducerTy = redxn.getReducerType();
  SmallVector<Value *, 2> args = redxn.getReducerArgs();
  CallInst *newCall = CallInst::Create(reducerTy, redxn.reducer, args);
  ReplaceInstWithInst(call, newCall);

  return true;
}

PreservedAnalyses LowerReduceIntrinsicsPass::run(Module &m,
                                                 ModuleAnalysisManager &am) {
  if (detail::lowerReduceIntrinsicsCore(m))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
