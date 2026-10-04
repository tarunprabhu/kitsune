//=- LowerReduceIntrinsics.h - Lower Kitsune's reduce intrinsics -*- C++ -*--=//
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

#ifndef KITSUNE_LIB_TRANSFORMS_LOWER_REDUCE_INTRINSICS_H
#define KITSUNE_LIB_TRANSFORMS_LOWER_REDUCE_INTRINSICS_H

namespace llvm {

class CallInst;

namespace detail {

class LowerGPUIntrImpl;

/// Lower a call to Kitsune's kit.reduce intrinsic with the 'naive' reduction
/// strategy. Always returns true.
bool lowerReduceIntrNaive(CallInst *call);

/// Lower a call to Kitsune's kit.reduce.0, or kit.gpu.reduce intrinsic. Always
/// returns true.
bool lowerReduceIntr(CallInst *call, LowerGPUIntrImpl &lowerGPU);

} // namespace detail

} // end namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_LOWER_REDUCE_INTRINSICS_H
