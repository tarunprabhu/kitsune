//===- TapirDebugUtils.cpp - Developer utilities for debugging ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions to render enums and types to make debugging easier and
// verbose mode for the tapir targets more helpful.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/TapirDebugUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

namespace llvm {

raw_ostream &operator<<(raw_ostream &os,
                        const FPOpFusion::FPOpFusionMode &mode) {
  switch (mode) {
  case FPOpFusion::Fast:
    return os << "fast";
  case FPOpFusion::Standard:
    return os << "standard";
  case FPOpFusion::Strict:
    return os << "strict";
  default:
    llvm_unreachable("operator<<: FPOpFusionMode not handled");
  }
}

raw_ostream &operator<<(raw_ostream &os, const OptimizationLevel &optLevel) {
  if (optLevel == OptimizationLevel::O0)
    return os << "O0";
  else if (optLevel == OptimizationLevel::O1)
    return os << "O1";
  else if (optLevel == OptimizationLevel::O2)
    return os << "O2";
  else if (optLevel == OptimizationLevel::O3)
    return os << "O3";
  else if (optLevel == OptimizationLevel::Os)
    return os << "Os";
  else if (optLevel == OptimizationLevel::Oz)
    return os << "Oz";
  else
    llvm_unreachable("operator:<< OptimizationLevel not handled");
}

raw_ostream &operator<<(raw_ostream &os,
                        const std::optional<TapirTargetID> &tt) {
  llvm::errs() << "trying (" << tt.has_value() << ")\n";
  if (tt.has_value())
    return os << *tt;
  else
    return os << "<<<std::nullopt>>>";
}

} // namespace llvm
