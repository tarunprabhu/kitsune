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

#include "LowerReduceIntrinsics.h"
#include "kitsune/Core/Reductions.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

bool llvm::detail::lowerReduce0Intr(CallInst *call) {
  ReductionInfo redxn(call);
  FunctionType *reducerTy = redxn.getReducerType();
  Value *reducer = redxn.getReducer();
  SmallVector<Value *, 2> args = redxn.getReducerArgs();
  CallInst *newCall = CallInst::Create(reducerTy, reducer, args);
  ReplaceInstWithInst(call, newCall);

  return true;
}
