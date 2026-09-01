//===- EmbLowerIntrinsics.cpp - Lower Kitsune-specific intrinsics ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune-specific intrinsics in an embedded module. Although most
// Kitsune-specific intrinsics are lowered by the backend, lowering of the
// GPU-specific intrinsics is run as part of Kitsune's lowering pipeline in the
// middle-end.
//
// In some cases, the intrinsics lower to LLVM-IR that can be optimized. In such
// cases, we don't strictly need to lower the intrinsics before codegen, but it
// would be sub-optimal to not do so. In other cases, the intrinsics *must* be
// lowered early. For example, when compiling for AMD GPU's, the standard set of
// optimization passes run on the embedded bitcode module includes the
// AMDGPUAttributor pass, which adds target-specific function attributes. This
// includes attributes such as `amdgpu-no-workgroup-id-y` that indicates that
// the function does not query workGroupId.y. There are corresponding attributes
// for `workitem`, `workGroupSize` etc. The attributor pass looks for the
// correct target-specific intrinsic calls to determine whether these parameters
// are queried. If these attributes are added, but the function does query these
// parameters, an invalid value will be returned at runtime.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EmbLowerIntrinsics.h"
#include "kitsune/Transforms/LowerReduceIntrinsics.h"
#include "LowerGPUIntrinsicsImpl.h"
#include "kitsune/Core/ModuleUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"

#define DEBUG_TYPE "emb-lower-intrinsics"

using namespace llvm;

// Not all intrinsics will be lowered here. Return true if it is lowered, false
// otherwise.
static bool lowerIntr(CallInst *call, detail::LowerGPUIntrImpl &lowerGPU) {
  switch (call->getIntrinsicID()) {
  case Intrinsic::kit_reduce_0: return detail::lowerReduce0Intr(call);
  case Intrinsic::kit_gpu_reduce_warp_shuffle_shared_memory:
    return lowerGPU.lowerReduceWarpShflShmemIntr(call);
  case Intrinsic::kit_gpu_reduce_shared_memory:
    return lowerGPU.lowerReduceWarpShflShmemIntr(call);
  case Intrinsic::kit_gpu_reduce_warp_shuffle:
    return lowerGPU.lowerReduceWarpShflIntr(call);
  case Intrinsic::kit_gpu_reduce_direct:
    return lowerGPU.lowerReduceDirectIntr(call);
  case Intrinsic::kit_gpu_warp_shfl_down_sync:
    return lowerGPU.lowerWarpShflDownSyncIntr(call);
  case Intrinsic::kit_gpu_warp_id: return lowerGPU.lowerWarpIdIntr(call);
  case Intrinsic::kit_gpu_warp_lane: return lowerGPU.lowerWarpLaneIntr(call);
  case Intrinsic::kit_gpu_warp_size: return lowerGPU.lowerWarpSizeIntr(call);
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
  case Intrinsic::kit_gpu_grid_size_z: return lowerGPU.lowerIndexIntr(call);
  default: break;
  }
  return false;
}

static bool lowerImpl(Module &m, detail::LowerGPUIntrImpl &lowerGPU) {
  SmallVector<CallInst *, 0> calls;
  for (Function &f : m)
    for (BasicBlock &bb : f)
      for (Instruction &inst : bb)
        if (auto *call = dyn_cast<CallInst>(&inst))
          if (Intrinsic::ID intr = call->getIntrinsicID())
            if (isKitIntrinsic(intr))
              calls.push_back(call);

  bool changed = false;
  for (CallInst *call : calls)
    changed |= lowerIntr(call, lowerGPU);
  return changed;
}

// The lowering of some intrinsics introduces calls to other intrinsics that
// are to be lowered by this pass. Iterate until no new intrinsics are
// generated.
static bool lowerIntrs(Module &m, detail::LowerGPUIntrImpl &lowerGPU) {
  bool result = false;
  bool changed = false;
  do {
    changed = lowerImpl(m, lowerGPU);
    result |= changed;
  } while (changed);
  return result;
}

static bool lowerIntrs(Module &m, const TTOptions &tto) {
  bool changed = false;
  detail::LowerGPUIntrImpl lowerGPU(tto);

  changed |= lowerIntrs(m, lowerGPU);

  return changed;
}

bool EmbLowerIntrinsicsPass::run(TTID tt, Module &devM, Module &hostM,
                                 ModuleAnalysisManager &hostAM) {
  return lowerIntrs(devM, tto);
}
