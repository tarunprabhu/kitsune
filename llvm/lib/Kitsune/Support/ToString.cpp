//=- ToString.cpp - String and serialization functions ------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implementation of additional string utilities and serialization functions.
///
//===----------------------------------------------------------------------===//

#include "kitsune/Support/ToString.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

std::string llvm::toString(const TTID &tt) {
  switch (tt) {
  case TTID::None:
    return "none";
  case TTID::Serial:
    return "serial";
  case TTID::Cuda:
    return "cuda";
  case TTID::Hip:
    return "hip";
  case TTID::Lambda:
    return "lambda";
  case TTID::OMPTask:
    return "omptask";
  case TTID::OpenCilk:
    return "opencilk";
  case TTID::OpenMP:
    return "openmp";
  case TTID::Qthreads:
    return "qthreads";
  case TTID::Realm:
    return "realm";
  }
  llvm_unreachable("toString: TTID not handled");
}

std::string llvm::toString(const MaybeBool &v) {
  switch (v) {
  case MaybeBool::Off:
    return "off";
  case MaybeBool::On:
    return "on";
  case MaybeBool::Any:
    return "any";
  }
  llvm_unreachable("toString: MaybeBool value not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const TTID &tt) {
  os << toString(tt);
  return os;
}

raw_ostream &llvm::operator<<(raw_ostream &os, const std::optional<TTID> &tt) {
  if (tt.has_value())
    return os << *tt;
  else
    return os << "<<<std::nullopt>>>";
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const TapirSpawnStrategy &strategy) {
  switch (strategy) {
  case TapirSpawnStrategy::Sequential:
    return os << "Sequential";
  case TapirSpawnStrategy::DivideAndConquer:
    return os << "Divide and conquer";
  case TapirSpawnStrategy::GPU:
    return os << "GPU";
  }
  llvm_unreachable("operator<<: TapirSpawnStrategy not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const MaybeBool &v) {
  os << toString(v);
  return os;
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const FPOpFusion::FPOpFusionMode &mode) {
  switch (mode) {
  case FPOpFusion::Fast:
    return os << "fast";
  case FPOpFusion::Standard:
    return os << "standard";
  case FPOpFusion::Strict:
    return os << "strict";
  }
  llvm_unreachable("operator<<: FPOpFusionMode not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const OptznLevel &optLevel) {
  switch (optLevel) {
  case OptznLevel::O0:
    return os << "O0";
  case OptznLevel::O1:
    return os << "O1";
  case OptznLevel::O2:
    return os << "O2";
  case OptznLevel::O3:
    return os << "O3";
  case OptznLevel::Os:
    return os << "Os";
  case OptznLevel::Oz:
    return os << "Os";
  }
  llvm_unreachable("operator<<: OptznLevel not handled");
}
