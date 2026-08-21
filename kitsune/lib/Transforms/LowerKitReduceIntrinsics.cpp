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
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/Reductions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "kit-reduce-intrinsics"

using namespace llvm;

// Collect all calls to the intrinsic \p id in a function.
static SmallVector<CallBase *, 0> collectCalls(Function &f, Intrinsic::ID id) {
  SmallVector<CallBase *, 0> calls;
  for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
    if (auto *call = dyn_cast<CallBase>(&*i))
      if (call->getIntrinsicID() == id)
        calls.push_back(call);
  return calls;
}

// The kit.reduce.0 intrinsics are replaced with a call to the reducer that
// was passed to it.
static bool lowerReduce0(Function &f) {
  LLVMContext &ctx = f.getContext();
  SmallVector<CallBase *, 0> calls = collectCalls(f, Intrinsic::kit_reduce_0);

  for (CallBase *call : calls) {
    ReductionInfo redxn(call);

    SmallVector<Value *, 2> args = {redxn.dest, redxn.value};
    args.append(redxn.getExtraArgs());

    Type *voidTy = Type::getVoidTy(ctx);
    SmallVector<Type *, 2> paramTys(args.size(), nullptr);
    for (unsigned i = 0; i < args.size(); ++i)
      paramTys[i] = args[i]->getType();

    FunctionType *fty = FunctionType::get(voidTy, paramTys, /*isVarArg=*/false);
    CallInst *newCall = CallInst::Create(fty, redxn.reducer, args);
    ReplaceInstWithInst(call, newCall);
  }

  return calls.size();
}

PreservedAnalyses
LowerKitReduceIntrinsicsPass::run(Function &f, FunctionAnalysisManager &am) {
  if (lowerReduce0(f))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
