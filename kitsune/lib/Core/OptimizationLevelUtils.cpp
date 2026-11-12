//===- OptimizationLevelUtils.cpp - Utilities for OptimizationLevel enum --===//
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

#include "kitsune/Core/OptimizationLevelUtils.h"
#include "llvm/Support/Error.h"

using namespace llvm;

template <> StringRef llvm::toString<OptimizationLevel>() {
  return "llvm::OptimizationLevel";
}

template <> std::string llvm::toString(const OptimizationLevel &optLevel) {
  switch (optLevel) {
  case OptimizationLevel::O0: return "O0";
  case OptimizationLevel::O1: return "O1";
  case OptimizationLevel::O2: return "O2";
  case OptimizationLevel::O3: return "O3";
  }
  llvm_unreachable("toString: OptimizationLevel not handled");
}

template <> std::optional<OptimizationLevel> llvm::fromInt(int64_t optLevel) {
  switch (optLevel) {
  case 0: return OptimizationLevel::O0;
  case 1: return OptimizationLevel::O1;
  case 2: return OptimizationLevel::O2;
  case 3: return OptimizationLevel::O3;
  default: break;
  }
  llvm_unreachable("fromInt: invalid speedup level");
}

OptimizationLevel llvm::createOptLevelFrom(char level) {
  switch (level) {
  case '0': return OptimizationLevel::O0;
  case '1': return OptimizationLevel::O1;
  case '2': return OptimizationLevel::O2;
  case '3': return OptimizationLevel::O3;
  default: break;
  }
  llvm_unreachable("createOptimizationLevelFrom: invalid optimization level");
}

CodeGenOptLevel llvm::createCodeGenOptLevelFrom(OptimizationLevel optLevel) {
  switch (optLevel) {
  case OptimizationLevel::O0: return CodeGenOptLevel::None;
  case OptimizationLevel::O1: return CodeGenOptLevel::Less;
  case OptimizationLevel::O2: return CodeGenOptLevel::Default;
  case OptimizationLevel::O3: return CodeGenOptLevel::Aggressive;
  }
  llvm_unreachable(
      "createCodeGenOptimizationLevelFrom: OptimizationLevel not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const OptimizationLevel &optLevel) {
  os << toString(optLevel);
  return os;
}
