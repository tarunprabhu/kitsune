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

using LowerFunc = bool(CallInst *call);

namespace {
struct LoweringInfo {
  unsigned priority;
  LowerFunc *lower;
};
} // namespace

// The lowering of some intrinsics will result in the introduction to calls to
// other intrinsics. Therefore, we need to lower them in order. Intrinsics with
// smaller priority values will be lowered before those with higher priority
// values. Intrinsics that have the same priority may be lowered in any order
// relative to one another. The value of 1024 is generally used for intrinsics
// that can be safely lowered after all other intrinsics in the map.
static const DenseMap<Intrinsic::ID, LoweringInfo> loweringInfo = {
    // Reduce intrinsics
    {Intrinsic::kit_gpu_reduce_warp_shuffle_shared_memory,
     {64, detail::lowerGPUReduceWarpShflShmemIntr}},
    {Intrinsic::kit_gpu_reduce_shared_memory,
     {68, detail::lowerGPUReduceShmemIntr}},
    {Intrinsic::kit_gpu_reduce_warp_shuffle,
     {72, detail::lowerGPUReduceWarpShflIntr}},
    {Intrinsic::kit_gpu_reduce_direct, {80, detail::lowerGPUReduceDirectIntr}},

    // Warp intrinsics
    {Intrinsic::kit_gpu_warp_shfl_down_sync,
     {128, detail::lowerGPUWarpShflDownSyncIntr}},
    {Intrinsic::kit_gpu_warp_id, {136, detail::lowerGPUWarpIdIntr}},
    {Intrinsic::kit_gpu_warp_lane, {136, detail::lowerGPUWarpLaneIntr}},
    {Intrinsic::kit_gpu_warp_size, {144, detail::lowerGPUWarpSizeIntr}},

    // Index intrinsics
    {Intrinsic::kit_gpu_thread_id_x, {1024, detail::lowerGPUIndexIntr}},
    {Intrinsic::kit_gpu_thread_id_y, {1024, detail::lowerGPUIndexIntr}},
    {Intrinsic::kit_gpu_thread_id_z, {1024, detail::lowerGPUIndexIntr}},
    {Intrinsic::kit_gpu_block_id_x, {1024, detail::lowerGPUIndexIntr}},
    {Intrinsic::kit_gpu_block_id_y, {1024, detail::lowerGPUIndexIntr}},
    {Intrinsic::kit_gpu_block_id_z, {1024, detail::lowerGPUIndexIntr}},
    {Intrinsic::kit_gpu_block_size_x, {1024, detail::lowerGPUIndexIntr}},
    {Intrinsic::kit_gpu_block_size_y, {1024, detail::lowerGPUIndexIntr}},
    {Intrinsic::kit_gpu_block_size_z, {1024, detail::lowerGPUIndexIntr}},
    {Intrinsic::kit_gpu_grid_size_x, {1024, detail::lowerGPUIndexIntr}},
    {Intrinsic::kit_gpu_grid_size_y, {1024, detail::lowerGPUIndexIntr}},
    {Intrinsic::kit_gpu_grid_size_z, {1024, detail::lowerGPUIndexIntr}},
};

static bool lowerIntrs(Module &m) {
  SmallVector<CallInst *, 0> calls;
  for (Function &f : m)
    for (BasicBlock &bb : f)
      for (Instruction &inst : bb)
        if (auto *call = dyn_cast<CallInst>(&inst))
          if (loweringInfo.contains(call->getIntrinsicID()))
            calls.push_back(call);

  std::sort(calls.begin(), calls.end(),
            [](const CallInst *l, const CallInst *r) -> bool {
              const LoweringInfo &li = loweringInfo.at(l->getIntrinsicID());
              const LoweringInfo &ri = loweringInfo.at(r->getIntrinsicID());
              return li.priority < ri.priority;
            });

  for (CallInst *call : calls)
    if (Intrinsic::ID id = call->getIntrinsicID())
      loweringInfo.at(id).lower(call);

  return calls.size();
}

bool EmbLowerIntrinsicsPass::run(TTID tt, Module &devM, Module &hostM,
                                 ModuleAnalysisManager &hostMAM) {
  return lowerIntrs(devM);
}
