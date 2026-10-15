//===- SCEVUtils.h - Utilities for SCEV objects ----------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for SCEV objects.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_SCEV_UTILS_H
#define KITSUNE_CORE_SCEV_UTILS_H

namespace llvm {

class SCEV;

/// Get the operand in the SCEV object \p sc that is not equal to \p match.
/// This requires \p sc to have exactly two operands. If neither operand matches
/// \p match, return nullptr.
const SCEV *getNonMatchingOperand(const SCEV *sc, const SCEV *match);

} // namespace llvm

#endif // KITSUNE_CORE_SCEV_UTILS_H
