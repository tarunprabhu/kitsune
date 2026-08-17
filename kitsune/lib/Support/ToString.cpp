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

template <> std::string llvm::toString(const bool &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const int8_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const uint8_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const int16_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const uint16_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const int32_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const uint32_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const int64_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const uint64_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const float &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const double &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const char *s) { return s; }

template <> std::string llvm::toString(const std::string &s) { return s; }

template <> std::string llvm::toString(const StringRef &s) { return s.str(); }
