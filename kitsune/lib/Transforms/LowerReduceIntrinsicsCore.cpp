//===- LowerReduceIntrinsicsCore.cpp - Lower core reduce intrinsics -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune's core reduction intrinsics.
//
//===----------------------------------------------------------------------===//

#include "LowerReduceIntrinsicsCore.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/Reductions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

bool llvm::detail::lowerReduceIntrinsicsCore(Module &m) {
  SmallVector<CallInst *, 0> calls = collectCalls(m, Intrinsic::kit_reduce_0);
  for (CallInst *call : calls) {
    ReductionInfo redxn(call);
    FunctionType *reducerTy = redxn.getReducerType();
    SmallVector<Value *, 2> args = redxn.getReducerArgs();
    CallInst *newCall = CallInst::Create(reducerTy, redxn.reducer, args);
    ReplaceInstWithInst(call, newCall);
  }
  return calls.size();
}
