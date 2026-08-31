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
#include "LowerGPUIntrinsicsImpl.h"
#include "kitsune/Core/ModuleUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"

#define DEBUG_TYPE "emb-lower-intrinsics"

using namespace llvm;

using LowerIntrImpl = detail::LowerGPUIntrImpl;
using LowerFunc = bool (LowerIntrImpl::*)(CallInst *);

static const DenseMap<Intrinsic::ID, LowerFunc> lowerFuncs = {
    // Reduce intrinsics
    {Intrinsic::kit_gpu_reduce_warp_shuffle_shared_memory,
     &LowerIntrImpl::lowerReduceWarpShflShmemIntr},
    {Intrinsic::kit_gpu_reduce_shared_memory,
     &LowerIntrImpl::lowerReduceShmemIntr},
    {Intrinsic::kit_gpu_reduce_warp_shuffle,
     &LowerIntrImpl::lowerReduceWarpShflIntr},
    {Intrinsic::kit_gpu_reduce_direct, &LowerIntrImpl::lowerReduceDirectIntr},

    // Warp intrinsics
    {Intrinsic::kit_gpu_warp_shfl_down_sync,
     &LowerIntrImpl::lowerWarpShflDownSyncIntr},
    {Intrinsic::kit_gpu_warp_id, &LowerIntrImpl::lowerWarpIdIntr},
    {Intrinsic::kit_gpu_warp_lane, &LowerIntrImpl::lowerWarpLaneIntr},
    {Intrinsic::kit_gpu_warp_size, &LowerIntrImpl::lowerWarpSizeIntr},

    // Index intrinsics
    {Intrinsic::kit_gpu_thread_id_x, &LowerIntrImpl::lowerIndexIntr},
    {Intrinsic::kit_gpu_thread_id_y, &LowerIntrImpl::lowerIndexIntr},
    {Intrinsic::kit_gpu_thread_id_z, &LowerIntrImpl::lowerIndexIntr},
    {Intrinsic::kit_gpu_block_id_x, &LowerIntrImpl::lowerIndexIntr},
    {Intrinsic::kit_gpu_block_id_y, &LowerIntrImpl::lowerIndexIntr},
    {Intrinsic::kit_gpu_block_id_z, &LowerIntrImpl::lowerIndexIntr},
    {Intrinsic::kit_gpu_block_size_x, &LowerIntrImpl::lowerIndexIntr},
    {Intrinsic::kit_gpu_block_size_y, &LowerIntrImpl::lowerIndexIntr},
    {Intrinsic::kit_gpu_block_size_z, &LowerIntrImpl::lowerIndexIntr},
    {Intrinsic::kit_gpu_grid_size_x, &LowerIntrImpl::lowerIndexIntr},
    {Intrinsic::kit_gpu_grid_size_y, &LowerIntrImpl::lowerIndexIntr},
    {Intrinsic::kit_gpu_grid_size_z, &LowerIntrImpl::lowerIndexIntr},
};

static bool lowerImpl(Module &m, LowerIntrImpl &lowerIntrImpl) {
  SmallVector<CallInst *, 0> calls;
  for (Function &f : m)
    for (BasicBlock &bb : f)
      for (Instruction &inst : bb)
        if (auto *call = dyn_cast<CallInst>(&inst))
          if (lowerFuncs.contains(call->getIntrinsicID()))
            calls.push_back(call);

  for (CallInst *call : calls) {
    if (Intrinsic::ID id = call->getIntrinsicID()) {
      LowerFunc lower = lowerFuncs.at(id);
      (lowerIntrImpl.*lower)(call);
    }
  }

  return calls.size();
}

static bool lowerIntrs(Module &m, const TTOptions &tto) {
  LowerIntrImpl lowerIntrImpl(tto);

  // The lowering of some intrinsics introduces calls to other intrinsics that
  // are to be lowered by this pass. Iterate until no new intrinsics are
  // generated.
  bool result = false;
  bool changed = false;
  do {
    changed = lowerImpl(m, lowerIntrImpl);
    result |= changed;
  } while (changed);

  return result;
}

bool EmbLowerIntrinsicsPass::run(TTID tt, Module &devM, Module &hostM,
                                 ModuleAnalysisManager &hostAM) {
  return lowerIntrs(devM, tto);
}
