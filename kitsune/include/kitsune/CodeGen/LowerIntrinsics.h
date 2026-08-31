//===- LowerIntrinsics.h - Lower Kitsune-specific intrinsics ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune-specific intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CODEGEN_LOWER_INTRINSICS_H
#define KITSUNE_CODEGEN_LOWER_INTRINSICS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class FunctionPass;

/// \ingroup kitsune
/// Lower Kitsune-specific intrinsics.
class LowerIntrinsicsPass : public PassInfoMixin<LowerIntrinsicsPass> {
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &am);
};

/// \ingroup kitsune
FunctionPass *createLowerIntrinsicsLegacyPass();

} // namespace llvm

#endif // KITSUNE_CODEGEN_LOWER_INTRINSICS_H
