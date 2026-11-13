//===- LowerIntrinsicsImpl.h - Lower Kitsune's intrinsics ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private implementation header to lower Kitsune's reduce intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_TRANSFORMS_LOWER_INTRINSICS_IMPL_H
#define KITSUNE_LIB_TRANSFORMS_LOWER_INTRINSICS_IMPL_H

namespace llvm {

class CallInst;
class TTOptions;

namespace detail {

// Lower a call to Kitsune's index intrinsic. These intrinsics correspond to
// the threadIdx, blockIdx, blockDim, gridSize, and gridDim builtin objects.
/// The call will be erased from the function. Always returns true.
bool lowerGPUIndexIntr(CallInst *call);

/// Lower a call to an intrinsic that will be lowered to an operation on a GPU
/// warp. The call will be erased from the function. Always returns true.
bool lowerGPUWarpIntr(CallInst *call, const TTOptions &tto);

/// Lower a call to Kitsune's kit.reduce intrinsic. The call will be erased from
/// the function. Always returns true.
bool lowerReduceIntr(CallInst *call);

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_LOWER_INTRINSICS_IMPL_H
