//===- TapirDebugUtils.cpp - Developer utilities for debugging ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions to render enums and types to string
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/TapirStringUtils.h"

namespace llvm {

raw_ostream &operator<<(raw_ostream &os, const TapirTargetID &tt) {
  switch (tt) {
  case TapirTargetID::None:
    return os << "none";
  case TapirTargetID::Serial:
    return os << "serial";
  case TapirTargetID::Cuda:
    return os << "cuda";
  case TapirTargetID::Hip:
    return os << "hip";
  case TapirTargetID::Lambda:
    return os << "lambda";
  case TapirTargetID::OMPTask:
    return os << "omptask";
  case TapirTargetID::OpenCilk:
    return os << "opencilk";
  case TapirTargetID::OpenMP:
    return os << "openmp";
  case TapirTargetID::Qthreads:
    return os << "qthreads";
  case TapirTargetID::Realm:
    return os << "realm";
  default:
    llvm_unreachable("Tapir target not handled");
  }
  return os;
}

raw_ostream &operator<<(raw_ostream &os, const TapirSpawnStrategy &strategy) {
  switch (strategy) {
  case TapirSpawnStrategy::Sequential:
    return os << "Sequential";
  case TapirSpawnStrategy::DivideAndConquer:
    return os << "Divide and conquer";
  case TapirSpawnStrategy::GPU:
    return os << "GPU";
  default:
    llvm_unreachable("operator<<: TapirSpawnStrategy not handled");
  }
}

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
  if (tt.has_value())
    return os << *tt;
  else
    return os << "<<<std::nullopt>>>";
}

raw_ostream &operator<<(raw_ostream &os, const std::optional<bool> &v) {
  // This prints 'on' and 'off' instead of true and false because 'on' and 'off'
  // are used in the command line options from which these are typically
  // obtained.
  if (not v.has_value())
    return os << "any";
  else if (*v)
    return os << "on";
  else
    return os << "off";
}

} // namespace llvm
