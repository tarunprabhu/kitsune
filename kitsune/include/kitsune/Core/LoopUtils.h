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

namespace llvm {

class BasicBlock;
class Function;
class LLVMContext;
class Loop;

/// \addtogroup kitsune
/// @{

/// Get the LLVMContext from the loop. This is only here because we cannot just
/// call loop.getContext(), but instead have to get the header before we can get
/// the context.
LLVMContext &getContext(Loop &loop);
LLVMContext &getContext(const Loop &loop);

/// Get the function containing the loop. If this is called while in the middle
/// of transforming a loop, it is possible that the containing function will
/// not be found. In this case, nullptr will be returned.
Function *getFunction(Loop &loop);
const Function *getFunction(const Loop &loop);

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

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_LOOP_UTILS_H
