//===- LowerIntrinsicsEarly.cpp - Early lowering of Kitsune's intrinsics --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune's intrinsics early, i.e. as part of the middle-end pass
// pipeline. Intrinsics are generally lowered as part of the codegen pipeline.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/LowerIntrinsicsEarly.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Passes/PassUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"

#define DEBUG_TYPE "kit-lower-intrinsics-early"

using namespace llvm;

static bool lowerItersPerThreadIntr(CallInst &call) {
  Value *arg = call.getArgOperand(1);
  assert((isa<GlobalVariable>(arg) &&
          cast<GlobalVariable>(arg)->hasInitializer()) &&
         "Argument to kit.gpu.ipt intrinsic must be a global variable with an "
         "initializer");

  Constant *ipt = cast<GlobalVariable>(arg)->getInitializer();
  if (auto *blksz = dyn_cast<ConstantInt>(ipt)) {
    call.replaceAllUsesWith(blksz);
    call.eraseFromParent();
    return true;
  } else {
    return emitDiagExit(
        call, DiagID::ErrNYI,
        "Computing iterations per thread for GPU loops at runtime");
  }
}

static bool maybeLowerIntrinsic(CallInst &call) {
  switch (call.getIntrinsicID()) {
  case Intrinsic::kit_gpu_ipt: return lowerItersPerThreadIntr(call);
  default: break;
  }
  return false;
}

static bool runOnFunction(Function &f) {
  SmallVector<CallInst *> calls;
  for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
    if (auto *call = dyn_cast<CallInst>(&*i))
      if (call->getIntrinsicID())
        calls.push_back(call);

  bool changed = false;
  for (CallInst *call : calls)
    changed |= maybeLowerIntrinsic(*call);
  return changed;
}

PreservedAnalyses LowerIntrinsicsEarlyPass::run(Function &f,
                                                FunctionAnalysisManager &am) {
  if (runOnFunction(f))
    return getPreservedAnalysesCallGraph();
  return getPreservedAnalysesAll();
}
