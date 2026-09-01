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

// Lower a call to Kitsune's kit.reduce.0 intrinsic. Always returns true.
bool lowerReduce0Intr(CallInst *call);

} // namespace detail

} // end namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_LOWER_REDUCE_INTRINSICS_H
