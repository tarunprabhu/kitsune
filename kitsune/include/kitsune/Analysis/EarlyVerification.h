//==- EarlyVerification.h - Kitsune-specific early verification --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kitsune-specific verification that is carried out early in the optimization
// pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_ANALYSIS_EARLY_VERIFICATION_H
#define KITSUNE_ANALYSIS_EARLY_VERIFICATION_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class LoopInfo;
class ScalarEvolutions;
class TaskInfo;

/// \addtogroup kitsune
/// @{

/// Kitsune-specific verification pass that runs early in the optimization
/// pipeline.
///
/// NOTE: This class is named "EarlyVerificationPass" even though
/// "EarlyVerifierPass" would be more consistent because LLVM disables
/// `-print-before` and `print-after` for passes whose name ends with
/// "VerifierPass" (among other "special" suffixes).
///
class EarlyVerificationPass : public PassInfoMixin<EarlyVerificationPass> {
private:
  bool exitIfError = true;

public:
  explicit EarlyVerificationPass(bool exitIfError = true)
      : exitIfError(exitIfError) {}

  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam);
};

/// @}

} // namespace llvm

#endif // KITSUNE_ANALYSIS_EARLY_VERIFICATION_H
