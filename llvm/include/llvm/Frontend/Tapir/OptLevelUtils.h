//==- OptLevelUtils.h - Utilities for optimization levels ----- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Helper functions to deal with optimization levels which tend to be passed
/// around as a number of different enums and even raw integer values.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TAPIR_OPT_LEVEL_UTILS_H
#define LLVM_TAPIR_OPT_LEVEL_UTILS_H

#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

/// Map a given optimization level to a CodeGenOptLevel.
CodeGenOptLevel mapToCodeGenOptLevel(OptimizationLevel OptLevel);

/// Map an integer to an optimization level. The integer must be in [0,3].
OptimizationLevel mapToOptimizationLevel(unsigned OptLevel);

} // namespace llvm

#endif // LLVM_TAPIR_OPT_LEVEL_UTILS_H
