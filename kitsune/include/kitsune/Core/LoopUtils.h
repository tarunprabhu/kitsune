//===- LoopUtils.h - Utilities for LLVM loops ------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM loops.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_LOOP_UTILS_H
#define KITSUNE_CORE_LOOP_UTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"

namespace llvm {

class BasicBlock;
class DominatorTree;
class Function;
class Instruction;
class LLVMContext;
class Loop;
class LoopInfo;
class Module;
class PHINode;
class Task;
class TaskInfo;

/// \addtogroup kitsune
/// @{

/// Get the LLVMContext from the loop. This is only here because we cannot just
/// call loop.getContext(), but instead have to get the header before we can get
/// the context.
LLVMContext &getContext(const Loop &loop);

/// Get the function containing the loop. If this is called while in the middle
/// of transforming a loop, it is possible that the containing function will
/// not be found. In this case, nullptr will be returned.
Function *getFunction(Loop &loop);
const Function *getFunction(const Loop &loop);

/// Get the module containing the loop, or nullptr if the function containing
/// the loop is not in a module.
Module *getModule(Loop &loop);
const Module *getModule(const Loop &loop);

/// Get the "name" of a loop. This will first check if a "loop.name" attribute
/// is present on for the loop and return the value of that attribute.
/// Otherwise, if \p useDebugInfo is `true` and debug information is available,
/// the "name" will be derived from it. This name will be of the form
/// "<file>:<line>:<col>". If the column number is not available, it will be of
/// the form "<file>:<line>". If either of the file name and line number are not
/// available, the debug information will not be used to compute the name.
/// If a name has not yet been determined, and the loop header basic block has a
/// name return that. Otherwise, return the default name.
std::string getName(const Loop &loop, StringRef defawlt = "<unnamed>");

/// Remove all attributes specific to tapir loops from the given loop.
void clearTapirLoopAttrs(Loop &loop);

/// Recursively gather all subloops of the given loop.
SmallVector<Loop *, 4> getAllSubLoops(Loop &loop);
SmallVector<const Loop *, 4> getAllSubLoops(const Loop &loop);

/// Get all the basic blocks in a loop that are not part of any subloops. For
/// instance, given the loop structure shown below (where the indentation
/// indicates the nesting of a loop):
///
///   i_1      // loop_i
///     j_1    // loop_j
///       k_1  // loop_k
///     j_2
///   i_2
///     l_1    // loop_l
///   i_3
///
/// this will return the following:
///
///   getBlocksNotInSubLoops(loop_i) == {i_1, i_2, i_3}
///   getBlocksNotInSubLoops(loop_j) == {j_1, j_2}
///   getBlocksNotInSubLoops(loop_k) == {k_1}
///   getBlocksNotInSubLoops(loop_l) == {l_1}
///
SmallVector<BasicBlock *, 8> getBlocksNotInSubLoops(const Loop &loop);

/// Get the unique backedge in the loop, if one exists.
BasicBlock *getUniqueBackEdge(const Loop &loop);

/// Return true if the loop is a tapir loop. This only checks if the loop has
/// the `tapir.loop.target` attribute.
bool isTapirLoop(const Loop &loop);

/// Return true if any of the ancestors of a loop are tapir loops. The given
/// loop is not required to be a tapir loop. If the given loop is a top-level
/// loop, return false.
bool isAnyAncestorTapirLoop(const Loop &loop);

/// Returns true if the loop is a tapir loop and ANY of the following conditions
/// hold:
///
///   - The loop is a top-level loop
///   - None of the ancestors of the loop are tapir loops
///
bool isTopLevelTapirLoop(const Loop &loop);

/// Returns true if the loop is a tapir loop and ALL of the following
/// conditions hold:
///
///   - The tapir.loop.target attribute of the loop is a GPU-centric tapir
///     target such as 'cuda', or 'hip'.
///
///   - Any tapir loops contained within it have the same tapir.loop.target
///     attribute.
///
/// No assumptions are made about the structure of the subloops. Any non-tapir
/// subloops are ignored.
///
bool isTapirLoopForGPU(const Loop &loop);

/// Returns true if the loop is a top-level tapir loop and isTapirLoopForGPU
/// returns true for the loop. See the documentation for isTapirLoopForGPU for
/// more details.
bool isTopLevelTapirLoopForGPU(const Loop &loop);

/// Get the nearest tapir loop that is an ancestor of \p loop. \p loop need not
/// be a tapir loop. The returned value is guaranteed to be distinct from
/// \p loop. Return nullptr if no such loop was found.
const Loop *getNearestAncestorTapirLoop(const Loop &loop);

/// Get the roots of all tapir loop nests in a function.
SmallVector<Loop *, 4> getTopLevelTapirLoops(LoopInfo &li);

/// Get all the tapir loops in a function.
SmallVector<Loop *, 4> getTapirLoops(LoopInfo &li);

/// Tapir loops may require mandatory LLVM loop attributes. Those can be added
/// using this function.
void addMandatoryLLVMLoopAttrs(Loop &loop);

/// Clear the mandatory LLVM loop attributes from the given tapir loop. \p loop
/// must be identifiable as a tapir loop.
void clearMandatoryLLVMLoopAttrs(Loop &loop);

/// Serialize the tapir loop \p loop. This will remove any tapir-specific
/// annotations from the loop in addition to serializing it, as well as any LLVM
/// loop annotations that were added due to it being a tapir loop. For instance,
/// `llvm.loop.unroll.disable` is added to all tapir loops. This will be
/// removed.
///
/// \param loop The tapir loop to be serialized
/// \param task The tapir task for \p loop
/// \param dt An optional dominator tree that will be updated if provided
/// \param ti An optional TaskInfo object that will be updated if provided
/// \param addSerializedAttr If true, the "tapir.loop.serialized" attribute will
///                          be added to the loop.
/// \return Always returns true
bool serializeTapirLoop(Loop &loop, Task &task, bool addSerializedAttr = true,
                        DominatorTree *dt = nullptr, TaskInfo *ti = nullptr);

/// Get the number of induction variables in the loop. This is the just the
/// count of the number of PHI nodes in the loop header.
unsigned getNumIndVars(const Loop &loop);

/// Check if the instruction \p inst is in the loop \p loop, or any of its
/// subloops. If \p strict is true, \p inst must be precisely in \p loop, and
/// not in a subloop of \p loop.
bool isInLoop(const Instruction &inst, const Loop &loop, LoopInfo &li,
              bool strict = false);

/// If the given induction variable has at least one use that is not in \p loop,
/// return true. \p iv is expected to be an induction variable associated with
/// \p loop.
bool isUsedOutsideLoop(const PHINode &iv, const Loop &loop, LoopInfo &li);

/// Get the sync region for a tapir loop. This simply looks at the terminator
/// of the tapir loop header, which is expected to be a detach instruction and
/// returns the syncregion associated with that instruction. It is an error if
/// the terminator of the loop header is not a detach instruction.
Value *getTapirLoopSyncRegion(Loop &loop);

/// Get the detach instruction for a tapir loop. This simply looks at the
/// terminator of the tapir loop header, which is expected to be a detach
/// instruction and returns that.
DetachInst *getTapirLoopDetachInst(Loop &loop);

/// Get the reattach instruction for a tapir loop. This simply looks at the
/// sole predecessor of the unique loop latch, whose terminator is expected to
/// be a reattach instruction.
ReattachInst *getTapirLoopReattachInst(Loop &loop);

/// Get the unique sync instruction for a tapir loop. This is the sole sync
/// instruction that is unconditionally reachable from the loop exit blocks. An
/// instruction is unconditionally reachable from an exit block if there exists
/// a unique path from the first instruction of the exit block to the sync
/// instruction. If more than one sync instruction is reachable from the loop's
/// exit blocks, returns nullptr.
SyncInst *getTapirLoopUniqueSyncInst(Loop &loop);

/// Get the "entry block" of the tapir loop body. This is the block that is
/// detached from the tapir loop header. For instance, for the tapir loop
/// below, this will return the basic block named "body".
///
///   header:
///     %i = phi i64 [ 0, %entry ], [ %next.i, %latch ]
///     detach within %syncreg, label %body, label %latch
///
///   body:
///     \<instructions\>
///
BasicBlock *getTapirLoopDetachedBlock(Loop &loop);

/// Get all blocks inside the loop that have successors outside the loop.
SmallVector<BasicBlock *, 2> getExitingBlocks(const Loop &loop);

/// Get all successor blocks of the loop.
SmallVector<BasicBlock *, 2> getExitBlocks(const Loop &loop);

/// If \p loop has exactly one exit block that is not a dead-end, return its
/// corresponding exiting block. If no such exiting block exists, return
/// nullptr.
BasicBlock *getUniqueNonDeadEndExitingBlock(const Loop &loop);

/// Get the unique exit (successor) block of a loop. These are blocks outside
/// the loop that are branched to from within the loop.
SmallVector<BasicBlock *, 2> getUniqueExitBlocks(const Loop &loop);

/// If \p loop has a unique exit block that is not a dead-end, return it,
/// otherwise, return nullptr.
BasicBlock *getUniqueNonDeadEndExitBlock(const Loop &loop);

/// Get the exit block of the loop that is the immediate successor of the loop
/// latch. This exit block may, or may not, be the unique loop exit block. In
/// general, the use of this function should be avoided in favor of
/// `Loop::getExitBlock()`, and should only be used when one knows that \p loop
/// does not have a unique exit block, but it does have a unique non-dead-end
/// exit block. One could use \ref getUniqueNonDeadEndExitBlock in such cases,
/// but this function is more efficient. If \p loop does not have a unique
/// latch, or if the terminator of the latch has more than two successors,
/// return nullptr.
BasicBlock *getExitBlockFromLatch(const Loop &loop);

/// Get a unique instruction of type \tparam InstType in a loop, or nullptr if
/// one does not exist. This will *not* look for such an instruction in any
/// subloops.
template <typename InstType>
static const InstType *getUniqueInstInLoopOnly(const Loop &loop) {
  const InstType *uniq = nullptr;
  for (const BasicBlock *bb : getBlocksNotInSubLoops(loop))
    for (const Instruction &inst : *bb)
      if (const auto *asType = dyn_cast<InstType>(&inst)) {
        // If an instruction of the given type has already been seen, a unique
        // instruction of the given type does not exist.
        if (uniq)
          return nullptr;
        uniq = asType;
      }

  // If no instructions of the given type were seen, this will be nullptr.
  // Otherwise, this will contain the unique instruction. If more than one
  // instruction of the given type was seen, the loop will have returned early -
  // control will never reach here.
  return uniq;
}

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_LOOP_UTILS_H
