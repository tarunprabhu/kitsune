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
#include "LowerGPUIntrinsicsImpl.h"
#include "kitsune/Core/Reductions.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

bool llvm::detail::lowerReduceIntrNaive(CallInst *call) {
  ReductionInfo redxn(call);
  assert(redxn.strategy == ReductionStrategy::Naive &&
         "Expected 'naive' reduction strategy");

  FunctionType *reducerTy = redxn.getReducerType();
  Value *reducer = redxn.getReducer();
  SmallVector<Value *, 2> args = redxn.getReducerArgs();
  CallInst *newCall = CallInst::Create(reducerTy, reducer, args);
  ReplaceInstWithInst(call, newCall);

  return true;
}

bool llvm::detail::lowerReduceIntr(CallInst *call,
                                   detail::LowerGPUIntrImpl &lowerGPU) {
  switch (ReductionInfo(call).strategy) {
  case ReductionStrategy::Auto:
    llvm_unreachable("Unexpected reduction strategy: 'auto'");
  case ReductionStrategy::Naive: return lowerReduceIntrNaive(call);
  case ReductionStrategy::ThreadLocal:
    llvm_unreachable("NOT IMPLEMENTED: ThreadLocal reduction strategy");
  case ReductionStrategy::Direct: return lowerGPU.lowerReduceIntrDirect(call);
  case ReductionStrategy::GPUSharedMemory:
    return lowerGPU.lowerReduceIntrShmem(call);
  case ReductionStrategy::GPUWarpShuffle:
    return lowerGPU.lowerReduceIntrWarpShfl(call);
  case ReductionStrategy::GPUWarpShuffleWithSharedMemory:
    return lowerGPU.lowerReduceIntrWarpShflShmem(call);
  default: break;
  }
  llvm_unreachable("lowerReduceIntr: ReductionStrategy not handled");
}
