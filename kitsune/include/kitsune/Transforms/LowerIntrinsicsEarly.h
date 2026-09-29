//===- LowerIntrinsicsEarly.h - Lower Kitsune's intrinsics -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune's intrinsics early, i.e. as part of the middle-end pass
// pipeline. Intrinsics are generally lowered as part of the codegen pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_LOWER_INTRINSICS_EARLY_H
#define KITSUNE_TRANSFORMS_LOWER_INTRINSICS_EARLY_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Lower Kitsune's intrinsics early, i.e. as part of the middle-end pass
/// pipeline.
class LowerIntrinsicsEarlyPass
    : public PassInfoMixin<LowerIntrinsicsEarlyPass> {
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &am);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_LOWER_INTRINSICS_EARLY_H
