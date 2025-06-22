//===-------- LLVM-provided High-Level Optimization levels -*- C++ -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header enumerates the LLVM-provided high-level optimization levels.
/// Each level has a specific goal and rationale.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_OPTIMIZATIONLEVEL_H
#define LLVM_PASSES_OPTIMIZATIONLEVEL_H

// The content of this file has been moved to
// llvm/Support/OptimizationLevel.h. The TapirTargetOptions object - defined
// in kitsune/Core - requires the OptimizationLevel that was originally defined
// here. This results in a circular dependence since KitCore => Passes =>
// TapirOpts => KitCore. Moving the OptimizationLevel class to LLVMSupport
// breaks this dependence. To avoid replacing all uses of this header in the
// rest of LLVM, we include llvm/Support/OptimizationLevel.h here instead. The
// corresponding source file has been moved to
// llvm/lib/Support/OptimizationLevel.cpp.

#include "llvm/Support/OptimizationLevel.h"

#endif // LLVM_PASSES_OPTIMIZATION_LEVEL_H
