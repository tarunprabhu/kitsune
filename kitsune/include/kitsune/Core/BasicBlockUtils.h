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

namespace llvm {

class BasicBlock;

/// \@addtogroup kitsune
/// @{

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

/// Return true if BOTH the following conditions hold:
///
///   - The basic block \p bb contains exactly one instruction
///   - That instruction is an UnreachableInst
///
/// NOTE: Despite what the name suggests, this DOES NOT have anything to do
/// with reachability. The predecessors of this basic block are not examined.
/// See also \ref isDisconnected and \ref isOrphaned.
bool isUnreachable(const BasicBlock &bb);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_BASIC_BLOCK_UTILS_H
