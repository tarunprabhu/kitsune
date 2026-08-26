//===- LowerKitReduceIntrinsics.cpp - Lower Kitsune's reduce intrinsics ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune's reduce intrinsics.
//
// Unlike most other intrinsics, this is done as part of the middle end. The
// reduction may involve generation of LLVM loops, or even tapir loops.
// Typically, several optimization passes will have to be run after this pass
// to ensure that any code that is generated here is optimized.
//
//
// kit.reduce.0 is reduced by simply calling the provided reducer function.
// For example, the call below
//
//   call void kit.reduce.0(i32 1, ptr %r, i32 4, i32 %v, i32 1, ptr @f, i8 %e)
//
// would become
//
//   call void @f(ptr %r, i32 %v, i8 %e)
//
// Note that here, %e is part of the optional arguments that the reducer
// function might accept.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/LowerKitReduceIntrinsics.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/Reductions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "kit-lower-reduce-intrinsics"

using namespace llvm;

// The kit.reduce.0 intrinsics are replaced with a call to the reducer that
// was passed to it.
static bool lowerReduce0(Module &m) {
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

PreservedAnalyses LowerKitReduceIntrinsicsPass::run(Module &m,
                                                    ModuleAnalysisManager &am) {
  if (lowerReduce0(m))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
