//===- TTObjectsAnalysis.h - Analysis pass for tapir targets ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass that determines the tapir targets needed by individual functions and
// loops in a module.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_ANALYSIS_TTOBJECTS_ANALYSIS_H
#define KITSUNE_ANALYSIS_TTOBJECTS_ANALYSIS_H

#include "kitsune/Core/TTID.h"
#include "kitsune/Core/TTOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

#include <map>

namespace llvm {

class LoopInfo;
class TapirTarget;
class TTObjectsAnalysis;
class TTObjectsAnalysisWrapperPass;
class TTOptions;
class TaskInfo;

/// \ingroup kitsune
/// An object that contains information about the tapir targets that are
/// enabled. This also owns the tapir target objects that are used when lowering
/// tapir constructs.
class TTObjects {
public:
  using GetLoopInfo = std::function<LoopInfo &(Function &)>;
  using GetTaskInfo = std::function<TaskInfo &(Function &)>;

private:
  /// Options for the primary tapir target.
  const TTOptions &ttOpts;

  /// The tapir targets used by each function in the module.
  std::map<Function *, SmallVector<TTID, 2>> ttsInFunc;

  /// The tapir targets needed by the functions in the module.
  SmallVector<TTID, 2> ttsInModule;

  /// The TapirTarget objects needed by the module. These will be created
  /// when \ref computeRequiredTTs is run. If computeRequiredTTs is run more
  /// than once, a new TapirTarget object will only be created if one does not
  /// already exist in this map. The actual TapirTarget objects are owned by the
  /// TTObjectsAnalysis object.
  std::map<TTID, TapirTarget *> tts;

private:
  TTObjects(const TTOptions &ttOpts);

  /// Compute the tapir target objects required by every function in a module.
  void computeRequiredTTs(Module &m, GetLoopInfo getLoopInfo,
                          GetTaskInfo getTaskInfo);

  /// Add a TapirTarget object for the given ID. The object will be owned by
  /// the TTObjectsAnalysis object.
  void addTT(TTID id, TapirTarget *tt);

  /// Get the tapir target options. This should only be called when the tapir
  /// target options are guaranteed to have been set.
  const TTOptions &getOptions() const { return ttOpts; }

  bool hasTTID() const { return ttOpts.hasTTID(); }

  /// Get the primary tapir target ID. This should only be called when the tapir
  /// target options are guaranteed to have been set.
  TTID getTTID() const;

public:
  /// Check if a TapirTarget exists for the given ID.
  bool hasTT(TTID id) const;

  /// Get the TapirTarget for the given ID. The id is assumed to exist.
  TapirTarget *getTT(TTID id) const;

  /// Get the tapir target ID's required by a function. This will be an empty
  /// vector if there are no tapir loops in the function. If there is at least
  /// one tapir loop in the function, this will contain the tapir target ID's
  /// that appear as loop hints on the loop (this is the case for attributed
  /// forall loops) and the primary tapir target ID if there is at least one
  /// tapir loop in the function which does not have a target loop hint.
  ArrayRef<TTID> getRequiredTTs(Function &f) const;

  /// Get the tapir target ID's required by the module. This will be an empty
  /// vector if there are no tapir loops in the module, even if a primary tapir
  /// target has been set by the frontend.
  ArrayRef<TTID> getRequiredTTs(Module &m) const;

  bool invalidate(Module &, const PreservedAnalyses &pa,
                  ModuleAnalysisManager::Invalidator &);

  friend class TTObjectsAnalysis;
  friend class TTObjectsAnalysisWrapperPass;
};

/// \ingroup kitsune
/// Analysis pass that contains TapirTarget instances required by loop-spawning
/// and any other passes that might need them. It can be used to query the
/// tapir targets that have been enabled explicitly (via the --tapir
/// command-line option passed to a compiler driver or the opt tool), as well
/// as the targets attached to individual tapir constructs.
class TTObjectsAnalysis : public AnalysisInfoMixin<TTObjectsAnalysis> {
private:
  /// The TTObjects instance that will be populated when @ref run() is called.
  /// A copy of this will be returned whenever the analysis is retrieved.
  TTObjects ttObjs;

  /// The tapir targets needed by the module.
  std::map<TTID, std::unique_ptr<TapirTarget>> tts;

public:
  using Result = TTObjects;

  /// Construct an analysis pass.
  TTObjectsAnalysis(const TTOptions &ttOpts);

  Result run(Module &m, ModuleAnalysisManager &mam);

private:
  static AnalysisKey Key;

  friend AnalysisInfoMixin<TTObjectsAnalysis>;
};

} // namespace llvm

#endif // KITSUNE_ANALYSIS_TTOBJECTS_ANALYSIS_H
