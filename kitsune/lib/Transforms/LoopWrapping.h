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

class DominatorTree;
class Loop;
class LoopInfo;
class MemorySSA;
class TapirLoopInfo;

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
                              LoopInfo &li);

/// Wrap the tapir loop \p loop with a serial loop.
///
/// The CFG of \p loop structure is expected to be as shown below. LoopPreheader
/// and LoopExit blocks are not part of the loop, so they are indented at a
/// different level.
///
///     LoopPreheader
///         LoopHeader
///         <LoopBlocks>
///         LoopLatch
///     LoopExit
///
/// The CFG after the loop is wrapped will be transformed as shown below.
///
///     LoopPreheader
///     OuterPreheader
///         OuterHeader
///         LoopGuardNew
///         LoopPreheaderNew
///             LoopHeader
///             <LoopBlocks>
///             LoopLatch
///         LoopExitNew
///         LoopEndNew
///         OuterReattach
///         OuterLatch
///     OuterExit
///     LoopExit
///
/// Here, the OuterReattach block is created to allow the caller a reasonable
/// place to insert a single reattach instruction if they intend to parallelize
/// the outer loop. Otherwise, the simplifycfg pass can be used to eliminate
/// the block.
///
/// NOTE: This will *NOT* change the bounds of either the original, nor the
/// newly created loops. It is the responsibility of the caller to ensure that
/// the transformation does not change the final behavior of the code.
///
/// The provided analysis objects, the dominator tree \p dt, the loop info
/// \p li, and the memory SSA \p mssa, will be updated and will remain valid
/// after this function returns.
///
/// Returns the newly created outer loop.
///
Loop *wrapWithTapirLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                        LoopInfo &li, MemorySSA &mssa);

} // namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_LOOP_WRAPPING_H
