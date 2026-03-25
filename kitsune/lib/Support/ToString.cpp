//===- ToString.cpp - String and serialization functions ------------------===//
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
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

template <> StringRef llvm::toString<int8_t>() { return "int8_t"; }
template <> StringRef llvm::toString<uint8_t>() { return "uint8_t"; }
template <> StringRef llvm::toString<int16_t>() { return "int16_t"; }
template <> StringRef llvm::toString<uint16_t>() { return "uint16_t"; }
template <> StringRef llvm::toString<int32_t>() { return "int32_t"; }
template <> StringRef llvm::toString<uint32_t>() { return "uint32_t"; }
template <> StringRef llvm::toString<int64_t>() { return "int64_t"; }
template <> StringRef llvm::toString<uint64_t>() { return "uint64_t"; }
template <> StringRef llvm::toString<float>() { return "float"; }
template <> StringRef llvm::toString<double>() { return "double"; }

template <> StringRef llvm::toString<std::string>() { return "std::string"; }
template <> StringRef llvm::toString<StringRef>() { return "llvm::StringRef"; }

template <> StringRef llvm::toString<TTID>() { return "llvm::TTID"; }
template <> StringRef llvm::toString<TapirSpawnStrategy>() {
  return "llvm::TapirSpawnStrategy";
}
template <> StringRef llvm::toString<OptznLevel>() {
  return "llvm::OptznLevel";
}

std::string llvm::toString(const TTID &tt) {
  switch (tt) {
  case TTID::Nolo:
    return "nolo";
  case TTID::Cuda:
    return "cuda";
  case TTID::Custom:
    return "custom";
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
  case TTID::Pthreads:
    return "pthreads";
  case TTID::Qthreads:
    return "qthreads";
  case TTID::Realm:
    return "realm";
  case TTID::Serial:
    return "serial";
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

std::string llvm::toString(const OptznLevel &optLevel) {
  switch (optLevel) {
  case OptznLevel::O0:
    return "O0";
  case OptznLevel::O1:
    return "O1";
  case OptznLevel::O2:
    return "O2";
  case OptznLevel::O3:
    return "O3";
  case OptznLevel::Os:
    return "Os";
  case OptznLevel::Oz:
    return "Os";
  }
  llvm_unreachable("operator<<: OptznLevel not handled");
}
