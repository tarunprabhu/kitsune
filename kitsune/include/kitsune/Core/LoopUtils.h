//===- LoopUtils.h - Utilities for LLVM loops ------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM Loop's.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_LOOP_UTILS_H
#define KITSUNE_CORE_LOOP_UTILS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class Loop;

/// The required prefix on the names of all tapir loop metadata.
static constexpr StringRef loopMDNamePrefix = "tapir.loop.";

/// @{
/// The names of the various hints. These are exactly they appear in the IR.
/// They *MUST* start with the prefix, tapir.loop.

constexpr StringRef loopMDNameStrategy = "tapir.loop.spawn.strategy";
constexpr StringRef loopMDNameGrainSize = "tapir.loop.grainsize";
constexpr StringRef loopMDNameLoopTarget = "tapir.loop.target";
constexpr StringRef loopMDNameThreadsPerBlock = "tapir.loop.threads.per.block";
constexpr StringRef loopMDNameAutotuneLaunch = "tapir.loop.autotune.launch";
constexpr StringRef loopMDNamePerfectDepth = "tapir.loop.perfect.depth";
constexpr StringRef loopMDNamePerfectLevel = "tapir.loop.perfect.level";

/// @}

/// Get the value of the tapir.loop.perfect.depth metadata on the given loop. If
/// the metadata does not exist on the loop, return 0.
unsigned getTapirLoopPerfectDepthMD(const Loop &loop);

/// Get the value of the tapir.loop.perfect.level metadata on the given loop. If
/// the metadata does not exist on the loop, return 0.
unsigned getTapirLoopPerfectLevelMD(const Loop &loop);

//// Set the tapir.loop.perfect.root metadata on the given loop. This should

/// Set the tapir.loop.perfect.depth metadata on the given loop. This should
/// only be set on the root of a "tapir loop nest". A tapir loop nest is a loop
/// nest where the root is a tapir loop. The value indicates the level of the
/// deepest tapir loop in the nest that is perfectly nested with respect to the
/// root. All loops between the deepest tapir loop and the root must also be
/// tapir loops.
void setTapirLoopPerfectDepthMD(Loop &loop, unsigned depth);

/// Set the tapir.loop.perfect.level metadata on the given loop. This should
/// only be set on subloops of a "tapir loop nest". A tapir loop nest is a loop
/// nest where the root is a tapir loop. The value indicates the level of the
/// deepest tapir loop in the nest that is perfectly nested with respect to the
/// root. All loops between the deepest tapir loop and the root must also be
/// tapir loops. Roots of tapir loop nests will have a level of 1.
void setTapirLoopPerfectLevelMD(Loop &loop, unsigned level);

} // namespace llvm

#endif // KITSUNE_CORE_LOOP_UTILS_H
