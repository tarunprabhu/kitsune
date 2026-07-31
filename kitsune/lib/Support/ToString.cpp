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

std::string llvm::toString(const bool &v) { return std::to_string(v); }

std::string llvm::toString(const int8_t &v) { return std::to_string(v); }

std::string llvm::toString(const uint8_t &v) { return std::to_string(v); }

std::string llvm::toString(const int16_t &v) { return std::to_string(v); }

std::string llvm::toString(const uint16_t &v) { return std::to_string(v); }

std::string llvm::toString(const int32_t &v) { return std::to_string(v); }

std::string llvm::toString(const uint32_t &v) { return std::to_string(v); }

std::string llvm::toString(const int64_t &v) { return std::to_string(v); }

std::string llvm::toString(const uint64_t &v) { return std::to_string(v); }

std::string llvm::toString(const float &v) { return std::to_string(v); }

std::string llvm::toString(const double &v) { return std::to_string(v); }

std::string llvm::toString(const char *s) { return s; }

std::string llvm::toString(const std::string &s) { return s; }

std::string llvm::toString(const StringRef &s) { return s.str(); }

std::string llvm::toString(const TTID &tt) {
  // clang-format off
  switch (tt) {
  case TTID::Nolo: return "nolo";
  case TTID::Cuda: return "cuda";
  case TTID::Custom: return "custom";
  case TTID::Hip: return "hip";
  case TTID::Lambda: return "lambda";
  case TTID::OMPTask: return "omptask";
  case TTID::OpenCilk: return "opencilk";
  case TTID::OpenMP: return "openmp";
  case TTID::Pthreads: return "pthreads";
  case TTID::Qthreads: return "qthreads";
  case TTID::Realm: return "realm";
  case TTID::Serial: return "serial";
  }
  // clang-format on
  llvm_unreachable("toString: TTID not handled");
}

std::string llvm::toString(const MaybeBool &v) {
  // clang-format off
  switch (v) {
  case MaybeBool::Off: return "off";
  case MaybeBool::On: return "on";
  case MaybeBool::Any: return "any";
  }
  // clang-format on
  llvm_unreachable("toString: MaybeBool value not handled");
}

std::string llvm::toString(const OptznLevel &optLevel) {
  // clang-format off
  switch (optLevel) {
  case OptznLevel::O0: return "O0";
  case OptznLevel::O1: return "O1";
  case OptznLevel::O2: return "O2";
  case OptznLevel::O3: return "O3";
  case OptznLevel::Os: return "Os";
  case OptznLevel::Oz: return "Os";
  }
  // clang-format on
  llvm_unreachable("toString: OptznLevel not handled");
}

std::string llvm::toString(const InstrumentKind &kind) {
  // clang-format off
  switch (kind) {
  case InstrumentKind::Generic: return "generic";
  case InstrumentKind::PAPI: return "papi";
  case InstrumentKind::Timer: return "timer";
  }
  // clang-format on
  llvm_unreachable("toString: InstrumentKind not handled");
}

std::string llvm::toString(const InstrumentUnit &unit) {
  // clang-format off
  switch (unit) {
  case InstrumentUnit::Thread: return "thread";
  case InstrumentUnit::Loop: return "loop";
  }
  // clang-format on
  llvm_unreachable("toString: InstrumentUnit not handled");
}
