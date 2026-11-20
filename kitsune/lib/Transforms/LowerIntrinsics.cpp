//===- LowerIntrinsics.cpp - Lower Kitsune's intrinsics in the middle-end -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private implementation to lower Kitsune's reduce intrinsics. This only
// handles a subset of the intrinsics that need to be lowered in the compiler
// middle-end. Most intrinsics are lowered during codegen.
//
//===----------------------------------------------------------------------===//

#include "LowerIntrinsics.h"
#include "kitsune/Core/IntrinsicUtils.h"
#include "kitsune/Core/TTUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

// This only lowers calls to @llvm.kit.cpu.num.threads with the serial tapir
// target. Calls to this intrinsic with other tapir targets will be lowered
// in a codegen pass.
static bool lowerNumThreadsIntrinsicSerial(CallInst *call) {
  assert(call->getIntrinsicID() == Intrinsic::kit_cpu_num_threads &&
         "Calls the CPU num threads intrinsic");
  assert(getTTIDFromKitIntrCall(*call) == TTID::Serial &&
         "Call to CPU num threads intrinsic with the serial tapir target");

  Type *ty = call->getType();
  Constant *one = ConstantInt::get(ty, 1, /*isSigned=*/false);

  call->replaceAllUsesWith(one);
  call->eraseFromParent();

  return true;
}

static detail::IntrinsicKind getIntrinsicKind(const CallInst &call) {
  std::optional<TTID> tt = getTTIDFromKitIntrCall(call);
  switch (call.getIntrinsicID()) {
  case Intrinsic::kit_cpu_num_threads:
    if (tt == TTID::Serial)
      return detail::IntrinsicKind::SerialNumThreads;
    break;
  case Intrinsic::kit_reduce:
    return isGPUTT(*tt) ? detail::IntrinsicKind::GPUReduce
                        : detail::IntrinsicKind::CPUReduce;
  case Intrinsic::kit_gpu_warp_id:
  case Intrinsic::kit_gpu_warp_lane:
  case Intrinsic::kit_gpu_warp_size:
  case Intrinsic::kit_gpu_warp_shfl_down_sync:
  case Intrinsic::kit_gpu_warp_sync:
    return detail::IntrinsicKind::GPUWarp;
  case Intrinsic::kit_gpu_thread_id_x:
  case Intrinsic::kit_gpu_thread_id_y:
  case Intrinsic::kit_gpu_thread_id_z:
  case Intrinsic::kit_gpu_block_id_x:
  case Intrinsic::kit_gpu_block_id_y:
  case Intrinsic::kit_gpu_block_id_z:
  case Intrinsic::kit_gpu_block_size_x:
  case Intrinsic::kit_gpu_block_size_y:
  case Intrinsic::kit_gpu_block_size_z:
  case Intrinsic::kit_gpu_grid_size_x:
  case Intrinsic::kit_gpu_grid_size_y:
  case Intrinsic::kit_gpu_grid_size_z: //
    return detail::IntrinsicKind::GPUIndex;
  default: break;
  }
  return detail::IntrinsicKind::Unknown;
}

bool detail::LowerIntrinsicsImpl::shouldLower(IntrinsicKind kind) const {
  return (kindsToLower & IntrinsicKinds(kind)) == IntrinsicKinds(kind);
}

bool detail::LowerIntrinsicsImpl::lowerIntrinsic(CallInst *call) {
  IntrinsicKind kind = getIntrinsicKind(*call);
  if (shouldLower(kind)) {
    switch (kind) {
    case detail::IntrinsicKind::SerialNumThreads:
      return lowerNumThreadsIntrinsicSerial(call);
    case detail::IntrinsicKind::CPUReduce:
    case detail::IntrinsicKind::GPUReduce: return lowerReduceIntrinsic(call);
    case detail::IntrinsicKind::GPUIndex: return lowerGPUIndexIntrinsic(call);
    case detail::IntrinsicKind::GPUWarp: return lowerGPUWarpIntrinsic(call);
    default: break;
    }
  }
  return false;
}

bool detail::LowerIntrinsicsImpl::runOn(Function &f) {
  bool result = false;
  bool changed = false;
  do {
    // The lowering of some intrinsics introduces calls to other intrinsics that
    // are to be lowered by this pass. Iterate until no new intrinsics are
    // generated.
    SmallVector<CallInst *, 0> calls;
    for (BasicBlock &bb : f)
      for (Instruction &inst : bb)
        if (auto *call = dyn_cast<CallInst>(&inst))
          if (Intrinsic::ID intr = call->getIntrinsicID())
            if (isKitIntrinsic(intr))
              calls.push_back(call);

    changed = false;
    for (CallInst *call : calls)
      changed |= lowerIntrinsic(call);

    result |= changed;
  } while (changed);
  return result;
}

bool detail::LowerIntrinsicsImpl::runOn(Module &m) {
  bool changed = false;
  for (Function &f : m)
    if (f.size())
      changed |= runOn(f);
  return changed;
}
