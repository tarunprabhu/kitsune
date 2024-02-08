//=- PreLowerVerification.h - Verification before tapir lowering --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kitsune-specific verification to be carried out just before tapir lowering.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_ANALYSIS_PRE_LOWER_VERIFICATION_H
#define KITSUNE_ANALYSIS_PRE_LOWER_VERIFICATION_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class LoopInfo;
class ScalarEvolutions;
class TaskInfo;

/// \addtogroup kitsune
/// @{

/// Kitsune-specific verification pass that runs just before tapir lowering.
/// Since this is a pass, it has access to the standard LLVM analyses (and
/// Kitsune-specific ones as well) that the standard verifier in
/// llvm/lib/IR/Verifier.cpp does not. As a result, it can perform more in-depth
/// analyses.
///
/// NOTE: This class is named "PreLowerVerificationPass" even though
/// "PreLowerVerifierPass" would be more consistent because LLVM disables
/// `-print-before` and `print-after` for passes whose name ends with
/// "VerifierPass" (among other "special" suffixes).
///
class PreLowerVerificationPass
    : public PassInfoMixin<PreLowerVerificationPass> {
private:
  bool exitIfError = true;

public:
  explicit PreLowerVerificationPass(bool exitIfError = true)
      : exitIfError(exitIfError) {}

  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam);
};

/// @}

} // namespace llvm

#endif // KITSUNE_ANALYSIS_PRE_LOWER_VERIFICATION_H
