//==- LowerGPUIntrinsicsImpl.h - Lower Kitsune's GPU intrinsics -*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune's GPU intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_TRANSFORMS_LOWER_GPU_INTRINSICS_IMPL_H
#define KITSUNE_LIB_TRANSFORMS_LOWER_GPU_INTRINSICS_IMPL_H

namespace llvm {

class CallInst;
class TTOptions;

namespace detail {

class LowerGPUIntrImpl {
protected:
  const TTOptions &tto;

public:
  LowerGPUIntrImpl(const TTOptions &tto) : tto(tto) {}

  // Lower a call to Kitsune's index intrinsic. These intrinsics correspond to
  // the threadIdx, blockIdx, blockDim, gridSize, and gridDim builtin objects.
  bool lowerIndexIntr(CallInst *call);

  // Lower a call to Kitsune's kit.gpu.reduce.direct intrinsic.
  bool lowerReduceDirectIntr(CallInst *call);

  // Lower a call to Kitsnue's kit.gpu.reduce.shared.memory intrinsic.
  bool lowerReduceShmemIntr(CallInst *call);

  // Lower a call to Kitsune's kit.gpu.reduce.warp.shuffle intrinsic.
  bool lowerReduceWarpShflIntr(CallInst *call);

  // Lower a call to Kitsune's kit.gpu.reduce.warp.shuffle.shared.memory
  // intrinsic.
  bool lowerReduceWarpShflShmemIntr(CallInst *call);

  // Lower a call to Kitsune's kit.gpu.warp.shfl.down.sync intrinsic.
  bool lowerWarpShflDownSyncIntr(CallInst *call);

  // Lower a call to Kitsune's kit.gpu.warp.id intrinsic.
  bool lowerWarpIdIntr(CallInst *call);

  // Lower a call to Kitsune's kit.gpu.warp.lane intrinsic.
  bool lowerWarpLaneIntr(CallInst *call);

  // Lower a call to Kitsune's kit.gpu.warp.size intrinsic.
  bool lowerWarpSizeIntr(CallInst *call);
};

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_LOWER_GPU_INTRINSICS_IMPL_H
