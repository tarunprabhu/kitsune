//===- Tapir.cpp - Developer utilities for debugging ----------------------===//
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

#include "llvm/Frontend/Tapir/Tapir.h"

namespace llvm {

std::string toString(const TapirTargetID &tt) {
  switch (tt) {
  case TapirTargetID::None:
    return "none";
  case TapirTargetID::Serial:
    return "serial";
  case TapirTargetID::Cuda:
    return "cuda";
  case TapirTargetID::Hip:
    return "hip";
  case TapirTargetID::Lambda:
    return "lambda";
  case TapirTargetID::OMPTask:
    return "omptask";
  case TapirTargetID::OpenCilk:
    return "opencilk";
  case TapirTargetID::OpenMP:
    return "openmp";
  case TapirTargetID::Qthreads:
    return "qthreads";
  case TapirTargetID::Realm:
    return "realm";
  default:
    llvm_unreachable("toString: TapirTargetID not handled");
  }
}

std::string toString(const MaybeBool &v) {
  switch (v) {
  case MaybeBool::Off:
    return "off";
  case MaybeBool::On:
    return "on";
  case MaybeBool::Any:
    return "any";
  default:
    llvm_unreachable("toString: MaybeBool value not handled");
  }
}

raw_ostream &operator<<(raw_ostream &os, const TapirTargetID &tt) {
  os << toString(tt);
  return os;
}

raw_ostream &operator<<(raw_ostream &os,
                        const std::optional<TapirTargetID> &tt) {
  if (tt.has_value())
    return os << *tt;
  else
    return os << "<<<std::nullopt>>>";
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

raw_ostream &operator<<(raw_ostream &os, const MaybeBool &v) {
  os << toString(v);
  return os;
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

} // namespace llvm
