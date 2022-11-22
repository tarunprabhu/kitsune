//===- LoopObjectsAnalysis.h - Objects used in a loop -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface to determine the objects that are used in
// a loop. This was originally developed for use by the Kitsune Cuda backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOPOBJECTS_ANALYSIS_H
#define LLVM_ANALYSIS_LOOPOBJECTS_ANALYSIS_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

#include <vector>

namespace llvm {

class TargetLibraryInfo;
class TaskInfo;

class LoopObjectsInfo {
public:
  LoopObjectsInfo(Loop *L, const TargetLibraryInfo *TLI,
                  TaskInfo *TI = nullptr);

  bool hasInfo() const;
  const std::vector<Value *> &getReadObjects() const;
  const std::vector<Value *> &getWriteObjects() const;

private:
  bool canAnalyzeLoop(TaskInfo *TI) const;
  void analyzeLoop(const TargetLibraryInfo *TLI);
  bool isAllocator(Function *Func, const TargetLibraryInfo *TLI);
  Value *trace(Value *V, const TargetLibraryInfo *TLI);

  template <typename I>
  std::vector<Value *> getUsedObjects(ArrayRef<BasicBlock *> BBS,
                                      const TargetLibraryInfo *TLI);

private:
  Loop *TheLoop;

  bool analyzed;
  std::vector<Value *> ReadObjects;
  std::vector<Value *> WriteObjects;
};

/// This analysis provides information about the objects that are accessed by
/// a loop.
///
/// It runs the analysis for a loop on demand.  This can be initiated by
/// querying the loop access info via AM.getResult<LoopObjectsAnalysis>.
/// getResult return a LoopObjectsInfo object.  See this class for the
/// specifics of what information is provided.
class LoopObjectsAnalysis : public AnalysisInfoMixin<LoopObjectsAnalysis> {
  friend AnalysisInfoMixin<LoopObjectsAnalysis>;
  static AnalysisKey Key;

public:
  typedef LoopObjectsInfo Result;

  Result run(Loop &L, LoopAnalysisManager &AM, LoopStandardAnalysisResults &AR);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_LOOPOBJECTS_ANALYSIS_H
