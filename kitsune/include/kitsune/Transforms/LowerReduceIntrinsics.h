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

#ifndef KITSUNE_TRANSFORMS_LOWER_REDUCE_INTRINSICS_H
#define KITSUNE_TRANSFORMS_LOWER_REDUCE_INTRINSICS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class CallInst;

namespace detail {

// Lower a call to Kitsune's kit.reduce.0 intrinsic. Always returns true.
bool lowerReduce0Intr(CallInst *call);

} // namespace detail

/// \ingroup kitsune
/// Lower Kitsune's reduce intrinsics. Instead of the lowering for most other
/// Kitsune intrinsics, this is intended to be run as part of the middle end.
/// The lowering may involve generation of LLVM loops, or even tapir loops, so
/// this should be run when other passes can optimize this lowered code.
class LowerReduceIntrinsicsPass
    : public PassInfoMixin<LowerReduceIntrinsicsPass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);
};

} // end namespace llvm

#endif // KITSUNE_TRANSFORMS_LOWER_REDUCE_INTRINSICS_H
