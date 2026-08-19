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
  bool changed = false;
  for (CallBase *call : collectCalls(f, Intrinsic::kit_reduce_0)) {
    Value *res = call->getArgOperand(2);
    Value *val = call->getArgOperand(4);
    Value *reducer = call->getArgOperand(6);

    SmallVector<Value *, 4> args = {res, val};
    for (unsigned i = 7; i < call->arg_size(); ++i)
      args.push_back(call->getArgOperand(i));

    Type *voidTy = Type::getVoidTy(ctx);
    SmallVector<Type *, 4> paramTys(args.size(), nullptr);
    for (unsigned i = 0; i < args.size(); ++i)
      paramTys[i] = args[i]->getType();

    FunctionType *fty = FunctionType::get(voidTy, paramTys, /*isVarArg=*/false);
    CallInst *newCall = CallInst::Create(fty, reducer, args);
    ReplaceInstWithInst(call, newCall);

    changed = true;
  }
  return changed;
}

PreservedAnalyses
LowerKitReduceIntrinsicsPass::run(Function &f, FunctionAnalysisManager &am) {
  DominatorTree &dt = am.getResult<DominatorTreeAnalysis>(f);
  LoopInfo &li = am.getResult<LoopAnalysis>(f);
  MemorySSA &mssa = am.getResult<MemorySSAAnalysis>(f).getMSSA();

  bool changed = lowerReduce0(f);

  if (changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
