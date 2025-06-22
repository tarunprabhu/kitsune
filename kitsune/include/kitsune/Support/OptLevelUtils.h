//=- OptLevelUtils.h - Utilities for LLVM's OptimizationLevel's ---*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's OptimizationLevel objects
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_OPT_LEVEL_UTILS_H
#define KITSUNE_SUPPORT_OPT_LEVEL_UTILS_H

#include "llvm/Support/CodeGen.h"
#include "llvm/Support/OptimizationLevel.h"

namespace llvm {

/// Map an optimization level to a CodeGenOptLevel.
CodeGenOptLevel mapToCodeGenOptLevel(OptimizationLevel optLevel);

/// Map an integer to an optimization level. The integer must be in [0,3].
OptimizationLevel mapToOptimizationLevel(unsigned level);

/// Map a character to an optimization level. The character must be in { '0',
/// '1', '2', '3', 's', 'z' }.
OptimizationLevel mapToOptimizationLevel(char level);

} // namespace llvm

#endif // KITSUNE_SUPPORT_OPT_LEVEL_UTILS_H
