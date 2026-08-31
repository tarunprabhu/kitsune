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

namespace detail {

// Lower a call to Kitsune's index intrinsic. These intrinsics correspond to
// the threadIdx, blockIdx, blockDim, gridSize, and gridDim builtin objects.
bool lowerGPUIndexIntr(CallInst *call);

// Lower a call to Kitsune's kit.gpu.reduce.direct intrinsic.
bool lowerGPUReduceDirectIntr(CallInst *call);

// Lower a call to Kitsnue's kit.gpu.reduce.shared.memory intrinsic.
bool lowerGPUReduceShmemIntr(CallInst *call);

// Lower a call to Kitsune's kit.gpu.reduce.warp.shuffle intrinsic.
bool lowerGPUReduceWarpShflIntr(CallInst *call);

// Lower a call to Kitsune's kit.gpu.reduce.warp.shuffle.shared.memory
// intrinsic.
bool lowerGPUReduceWarpShflShmemIntr(CallInst *call);

// Lower a call to Kitsune's kit.gpu.warp.shfl.down.sync intrinsic.
bool lowerGPUWarpShflDownSyncIntr(CallInst *call);

// Lower a call to Kitsune's kit.gpu.warp.id intrinsic.
bool lowerGPUWarpIdIntr(CallInst *call);

// Lower a call to Kitsune's kit.gpu.warp.lane intrinsic.
bool lowerGPUWarpLaneIntr(CallInst *call);

// Lower a call to Kitsune's kit.gpu.warp.size intrinsic.
bool lowerGPUWarpSizeIntr(CallInst *call);

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_LOWER_GPU_INTRINSICS_IMPL_H
