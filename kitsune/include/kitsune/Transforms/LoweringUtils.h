//===- LoweringUtils.h - Utilities for lowering tapir lopops ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions and class for lowering tapir loops. This is a wrapper
// around llvm/Transforms/Tapir/LoweringUtils.h and provides both the core Tapir
// lowering utilities, and other, Kitsune-specific, utilities. Especially in
// Kitsune-specific code, this should be included instead of including
// Tapir/LoweringUtils.h
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_LOWERING_UTILS_H
#define KITSUNE_TRANSFORMS_LOWERING_UTILS_H

#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {

class AssumptionCache;
class DominatorTree;
class LoopInfo;
class OptimizationRemarkEmitter;
class ScalarEvolution;
class TargetLibraryInfo;
class TargetTransformInfo;
class TaskInfo;

/// A wrapper object containing analyses that are required by tapir targets
/// and loop outline processors. An instance of this will be owned by the
/// loop-spawning pass and should be provide to consumers as needed. This object
/// may also be used by the tapir targets, but, at the time of writing, the
/// tapir target objects live longer than the loop-spawning pass, so it is not
/// clear how this might be implemented.
///
/// NOTE: In general, managing analyses and tapir target objects is a
/// complicated mess. It is not clear how we should handle this.
///
/// WARNING: The analyses provided here must be used with care. Currently,
/// loop-spawning does not update the analyses after a loop has been processed.
/// As a result, the analysis objects may not be valid. In general, it is only
/// safe to query loop-local properties for loops that have not yet been
/// transformed. At some point, loop-spawning will be fixed so the analyses
/// are always valid. At that point, this warning should be removed.
struct FunctionAnalyses {
  AssumptionCache &ac;
  DominatorTree &dt;
  LoopInfo &li;
  OptimizationRemarkEmitter &ore;
  ScalarEvolution &se;
  TargetLibraryInfo &tli;
  TargetTransformInfo &tti;
  TaskInfo &ti;

  FunctionAnalyses(AssumptionCache &ac, DominatorTree &dt, LoopInfo &li,
                   OptimizationRemarkEmitter &ore, ScalarEvolution &se,
                   TargetLibraryInfo &tli, TargetTransformInfo &tti,
                   TaskInfo &ti)
      : ac(ac), dt(dt), li(li), ore(ore), se(se), tli(tli), tti(tti), ti(ti) {}
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_LOWERING_UTILS_H
