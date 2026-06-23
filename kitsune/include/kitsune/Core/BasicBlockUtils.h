//===- BasicBlockUtils.h - Utilities for LLVM's Basic Blocks ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's Basic Blocks.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_BASIC_BLOCK_UTILS_H
#define KITSUNE_CORE_BASIC_BLOCK_UTILS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class BasicBlock;
class Module;

/// \@addtogroup kitsune
/// @{

/// Get the module containing the function that contains a basic block, or
/// nullptr, if the basic block is not in a function, or if the function
/// containing it is not in a module.
Module *getModule(BasicBlock &bb);
const Module *getModule(const BasicBlock &bb);

/// Get the name of a basic block. If the basic block is unnamed, a string of
/// the form `%<N>` will be returned. This is how the basic block might appear
/// in human-readable LLVM-IR.
std::string getName(const BasicBlock &bb);

/// Return true if BOTH the following conditions hold:
///
///   - The basic block \p bb has no predecessors
///   - \p bb has no successors
///
/// See also \ref isOrphaned
bool isDisconnected(const BasicBlock &bb);

/// Return true if the basic block \p bb has no predecessors. It may or may not
/// have any successors. See also \ref isDisconnected
bool isOrphaned(const BasicBlock &bb);

/// Return true if the basic block \p bb is a dead-end. \p bb is a dead-end if
/// either of the following is true:
///
///   - The terminator of \p bb is an UnreachableInst.
///
///   - The terminator of \p bb is an unconditional branch, and the sole
///     successor of \p bb is a dead-end.
///
bool isDeadEnd(const BasicBlock &bb);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_BASIC_BLOCK_UTILS_H
