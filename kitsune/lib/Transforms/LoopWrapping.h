//==- LoopWrapping.h - Utilities to wrap loops with other loops -*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The utilities here wrap tapir loops with other loops. These are essentially
// helpers for other transformations that are similar to tiling/strip-mining but
// are very specific to tapir loops.
//
// These are also unusual in that they intentionally do not guarantee that the
// resulting loop nest will preserve the behavior of the original loop. It is
// up to the caller to adjust trip counts appropriately.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_TRANSFORMS_LOOP_WRAPPING_H
#define KITSUNE_LIB_TRANSFORMS_LOOP_WRAPPING_H

namespace llvm {

class BranchInst;
class DominatorTree;
class Loop;
class LoopInfo;
class MemorySSA;
class ScalarEvolution;
class TapirLoopInfo;
class TaskInfo;

/// Check that the given tapir loop can be wrapped in another loop. The checks
/// here are fairly strict. There are two main reasons for this:
///
///   - We want to simplify the wrapping functions
///   - It becomes difficult to check that the CFG of the wrapped loop is
///     correct if some of the conditions checked here are violated.
///
/// This will emit an error message and return false if at least one error was
/// found, true otherwise.
bool checkTapirLoopSafeToWrap(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                              LoopInfo &li, ScalarEvolution &se);

/// Wrap the tapir loop \p tapirLoop with a serial loop.
///
/// The CFG of \p tapirLoop is expected to be as shown below. If this is not the
/// case, the code will fail - either gracefully with an error message, but
/// potentially catastrophically. Since this is really intended to be used for
/// mandatory transformations of tapir loops, we prefer to fail early.
///
///     LoopGuard
///     LoopPreheader
///         LoopHeader
///         <LoopBlocks>
///         LoopLatch
///     LoopExit
///     LoopBypass
///
/// We impose additional constraints on the blocks themselves. The LoopBypass
/// block is expected to contain a single sync instruction, and nothing else.
/// Similarly, the LoopExit block is expected to be empty, with an unconditional
/// branch to LoopBypass.
///
/// The CFG after the loop is wrapped will be transformed as shown below.
///
///     LoopGuard1
///     OuterGuard
///     OuterPreheader
///         OuterHeader
///         LoopGuard2
///         LoopPreheader
///             LoopHeader
///             <LoopBlocks>
///             LoopLatch
///         LoopExit
///         LoopBypass
///         OuterReattach
///         OuterLatch
///     OuterExit
///     OuterBypass
///
/// Here, the original loop guard branch is split into two. In the most extreme,
/// case everything except the terminator will become part of LoopGuard1.
/// LoopGuard2, then, will only contain the terminator of the original LoopGuard
/// block (this is expected to be a conditional branch instruction that will
/// branch to either LoopPreheader or LoopBypass). However, we try to keep the
/// branch and its condition in LoopGuard2 if possible. By contrast, the
/// LoopBypass block remains exactly as it was before the transformation, except
/// that the sync instruction that terminated it now branches to OuterReattach.
///
/// The OuterReattach block is an artifact of the implementation - which is
/// shared by this function and \ref wrapWithTapirLoop. In this case, it will
/// be an empty block terminating in an unconditional branch to OuterLatch.
/// However, when using \ref wrapWithTapirLoop, this block will terminate in a
/// ReattachInst.
///
/// IMPORTANT: The initial value of the wrapper loop induction variable is 0.
/// The step is 1 and the final value is 0. It is up to the caller to change the
/// bounds.
///
/// The provided analysis objects, loop info \p li, memory SSA \p mssa, task
/// info \ti, and the dominator tree \p dt will be updated and will remain valid
/// after this function returns.
///
/// Returns the newly created outer loop.
///
Loop *wrapWithSerialLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                         LoopInfo &li, MemorySSA &mssa, ScalarEvolution &se,
                         TaskInfo &ti);

/// Wrap the tapir loop \p tapirLoop with a tapir loop.
///
/// See the documentation of \ref wrapWithSerialLoop for more information.
Loop *wrapWithTapirLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                        LoopInfo &li, MemorySSA &mssa, ScalarEvolution &se,
                        TaskInfo &ti);

} // namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_LOOP_WRAPPING_H
