//===- LowerKitsuneRuntimeIntrinsics.h - Lower kitrt intrinsics -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers intrinsics that correspond to functions in Kitsune's
// runtime.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_KITSUNE_LOWER_KITSUNE_RUNTIME_INTRINSICS_H
#define LLVM_TRANSFORMS_KITSUNE_LOWER_KITSUNE_RUNTIME_INTRINSICS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// This pass is responsible for lowering Kitsune's runtime intrinsics. These
/// are intrinsics that correspond to a function in kitsune's runtime.
class LowerKitsuneRuntimeIntrinsicsPass
    : public PassInfoMixin<LowerKitsuneRuntimeIntrinsicsPass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam);

  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_KITSUNE_LOWER_KITSUNE_RUNTIME_INTRINSICS_H
