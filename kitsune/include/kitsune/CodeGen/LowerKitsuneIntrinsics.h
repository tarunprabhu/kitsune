//=- LowerKitsuneIntrinsics.h - Lower Kitsune-specific intrinsics -*- C++ -*-=//
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

#ifndef KITSUNE_CODEGEN_LOWER_KITSUNE_INTRINSICS_H
#define KITSUNE_CODEGEN_LOWER_KITSUNE_INTRINSICS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class ModulePass;

/// \ingroup kitsune
/// Lower Kitsune-specific intrinsics.
class LowerKitsuneIntrinsicsPass
    : public PassInfoMixin<LowerKitsuneIntrinsicsPass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);

  static bool isRequired() { return true; }
};

/// \ingroup kitsune
ModulePass *createLowerKitsuneIntrinsicsLegacyPass();

} // namespace llvm

#endif // KITSUNE_CODEGEN_LOWER_KITSUNE_INTRINSICS_H
