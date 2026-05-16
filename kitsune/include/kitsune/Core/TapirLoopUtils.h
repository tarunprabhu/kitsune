//===- TapirLoopUtils.h - Utilities for tapir loops ------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for tapir loops.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_TAPIR_LOOP_UTILS_H
#define KITSUNE_CORE_TAPIR_LOOP_UTILS_H

namespace llvm {

class Instruction;
class PHINode;
class TapirLoopInfo;

/// \addtogroup kitsune
/// \@{

/// Get the primary induction variable for a tapir loop.
PHINode *getPrimaryIV(const TapirLoopInfo &loop);

/// Get the increment instruction for the primary induction variable of a tapir
/// loop. The tapir loop is assumed to have a unique latch. This is the case
/// when the loop is in simplify form, but that is not strictly required to use
/// this utility.
Instruction *getPrimaryIVInc(const TapirLoopInfo &loop);

/// \@}

} // namespace llvm

#endif // KITSUNE_CORE_TAPIR_LOOP_UTILS_H
