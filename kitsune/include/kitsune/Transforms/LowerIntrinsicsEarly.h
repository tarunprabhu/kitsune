//===- LowerIntrinsicsEarly.h - Lower Kitsune's intrinsics ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune's intrinsics in the middle-end. This only lowers a subset of
// intrinsics that must be lowered before codegen.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_LOWER_INTRINSICS_EARLY_H
#define KITSUNE_TRANSFORMS_LOWER_INTRINSICS_EARLY_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class TTOptions;

/// \ingroup kitsune
/// Lower Kitsune's intrinsics in the middle-end.
class LowerIntrinsicsEarlyPass
    : public PassInfoMixin<LowerIntrinsicsEarlyPass> {
private:
  const TTOptions &tto;

public:
  LowerIntrinsicsEarlyPass(const TTOptions &tto) : tto(tto) {}

  PreservedAnalyses run(Function &f, FunctionAnalysisManager &am);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_LOWER_INTRINSICS_EARLY_H
