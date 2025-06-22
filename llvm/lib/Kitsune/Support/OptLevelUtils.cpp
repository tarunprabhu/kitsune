//===- OptLevelUtils.cpp - Utilities for LLVM's OptimizationLevel's -------===//
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

#include "kitsune/Support/OptLevelUtils.h"
#include "llvm/Support/Error.h"

using namespace llvm;

CodeGenOptLevel llvm::mapToCodeGenOptLevel(OptimizationLevel optLevel) {
  switch (optLevel.getSpeedupLevel()) {
  case 0:
    return CodeGenOptLevel::None;
  case 1:
    return CodeGenOptLevel::Less;
  case 2:
    return CodeGenOptLevel::Default;
  case 3:
    return CodeGenOptLevel::Aggressive;
  default:
    llvm_unreachable("mapToCodeGenOptLevel: unknown speedup level");
  }
}

OptimizationLevel llvm::mapToOptimizationLevel(unsigned level) {
  switch (level) {
  case 0:
    return OptimizationLevel::O0;
  case 1:
    return OptimizationLevel::O1;
  case 2:
    return OptimizationLevel::O2;
  case 3:
    return OptimizationLevel::O3;
  default:
    llvm_unreachable("mapToOptimizationLevel: invalid optimization level");
  }
}

OptimizationLevel llvm::mapToOptimizationLevel(char level) {
  switch (level) {
  case '0':
    return OptimizationLevel::O0;
  case '1':
    return OptimizationLevel::O1;
  case '2':
    return OptimizationLevel::O2;
  case '3':
    return OptimizationLevel::O3;
  case 's':
    return OptimizationLevel::Os;
  case 'z':
    return OptimizationLevel::Oz;
  default:
    llvm_unreachable("mapToOptimizationLevel: unknown optimization level");
  }
}
