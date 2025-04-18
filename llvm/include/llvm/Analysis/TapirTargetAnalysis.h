//===- llvm/Analysis/TapirTargetAnalysis.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An interface for information about the tapir targets needed by a module.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_TAPIR_TARGET_ANALYSIS_H
#define LLVM_ANALYSIS_TAPIR_TARGET_ANALYSIS_H

#include "llvm/Frontend/Tapir/TapirTargetOptions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Tapir/TapirTargets.h"

#include <optional>

namespace llvm {

class TapirTargetAnalysis;

/// An object that contains information about the TapirTargets that are
/// enabled. Currently, it only has information about the "primary" tapir target
/// but it should be extended to include information about the
/// "construct-specific" tapir targets that may be used. The primary tapir
/// target and associated options are computed exactly once - either when the
/// pass is instantiated, or the first time the analysis results are requested.
/// The construct-specific information will be computed from the module for
/// which the analysis is requested and may be recomputed as needed.
class TapirTargetInfo {
private:
  /// The options for the primary tapir target. If not null, the pointee is
  /// owned by the TapirTargetAnalysis pass and must not be freed.
  const TapirTargetOptions *ttOpts;

private:
  TapirTargetInfo(const TapirTargetOptions *ttOpts) : ttOpts(ttOpts) {}

public:
  TapirTargetInfo() = delete;

  bool hasID() const { return ttOpts; }

  TapirTargetID getID() const {
    assert(ttOpts && "Tapir target options have not been set");
    return ttOpts->getTapirTargetID();
  }

  std::optional<TapirTargetID> getIDIfAvailable() const {
    if (ttOpts)
      return ttOpts->getTapirTargetID();
    return std::nullopt;
  }

  const TapirTargetOptions &getOptions() const {
    assert(ttOpts && "Tapir target options have not been set");
    return *ttOpts;
  }

  bool invalidate(Module &, const PreservedAnalyses &,
                  ModuleAnalysisManager::Invalidator &) {
    // The TapirTargetInfo is immutable for a module.
    return false;
  }

  friend class TapirTargetAnalysis;
};

/// Analysis pass to provide information about the "global" tapir targets.
/// Individual loops and other constructs may require other tapir targets.
/// Eventually, this should probably contain information about all the tapir
/// targets that are needed by a given module, but that has not yet been
/// implemented.
class TapirTargetAnalysis : public AnalysisInfoMixin<TapirTargetAnalysis> {
  friend AnalysisInfoMixin<TapirTargetAnalysis>;

  static AnalysisKey Key;

private:
  /// The primary tapir target options. This will be nullptr if a tapir target
  /// was not provided to the frontend.
  std::unique_ptr<TapirTargetOptions> ttOpts = nullptr;

public:
  using Result = TapirTargetInfo;

  TapirTargetAnalysis() = delete;

  /// Construct an analysis pass with an optional TapirTargetOptions object.
  /// This may be std::nullopt if the frontend creating this pass has not been
  /// given a tapir target to use.
  TapirTargetAnalysis(std::optional<TapirTargetOptions> ttOpts);

  Result run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_TAPIR_TARGET_ANALYSIS_H
