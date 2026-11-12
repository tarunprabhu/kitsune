//==- OptimizationLevelUtils.h - Utilities for OptimizationLevel -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's OptimizationLevel enum.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_OPTIMIZATION_LEVEL_UTILS_H
#define KITSUNE_CORE_OPTIMIZATION_LEVEL_UTILS_H

#include "kitsune/Support/FromInt.h"
#include "kitsune/Support/ToString.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

class raw_ostream;

/// Stream an OptimizationLevel using LLVM's streams.
raw_ostream &operator<<(raw_ostream &os, const OptimizationLevel &v);

/// Map a character to an \ref OptimizationLevel. The character must be one of
/// '0', '1', '2', or '3'.
OptimizationLevel createOptLevelFrom(char level);

/// Map an optimization level to a CodeGenOptimizationLevel.
CodeGenOptLevel createCodeGenOptLevelFrom(OptimizationLevel optLevel);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_OPTIMIZATION_LEVEL_UTILS_H
