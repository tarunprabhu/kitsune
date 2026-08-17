//===- TTID.cpp - The core TTID enum --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define the core TTID enum for the supported tapir targets.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TTID.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"

using namespace llvm;

template <> StringRef llvm::toString<TTID>() { return "llvm::TTID"; }

template <> std::string llvm::toString(const TTID &tt) {
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
  llvm_unreachable("toString: TTID not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const TTID &tt) {
  os << toString(tt);
  return os;
}

template <> std::optional<TTID> llvm::fromString(StringRef s) {
  return StringSwitch<std::optional<TTID>>(s)
      .Case("nolo", TTID::Nolo)
      .Case("cuda", TTID::Cuda)
      .Case("custom", TTID::Custom)
      .Case("hip", TTID::Hip)
      .Case("lambda", TTID::Lambda)
      .Case("omptask", TTID::OMPTask)
      .Case("opencilk", TTID::OpenCilk)
      .Case("openmp", TTID::OpenMP)
      .Case("pthreads", TTID::Pthreads)
      .Case("qthreads", TTID::Qthreads)
      .Case("realm", TTID::Realm)
      .Case("serial", TTID::Serial)
      .Default(std::nullopt);
}

template <> std::optional<TTID> llvm::fromInt(int64_t v) {
  switch (v) {
  case 0x0: return TTID::Nolo;
  case 0x1: return TTID::Serial;
  case 0x2: return TTID::Cuda;
  case 0x4: return TTID::Hip;
  case 0x8: return TTID::OpenCilk;
  case 0x20: return TTID::Qthreads;
  case 0x40: return TTID::Realm;
  case 0x80: return TTID::Lambda;
  case 0x100: return TTID::OMPTask;
  case 0x200: return TTID::OpenMP;
  case 0x400: return TTID::Pthreads;
  case 0x800: return TTID::Custom;
  default: return std::nullopt;
  }
}

raw_ostream &llvm::operator<<(raw_ostream &os, const std::optional<TTID> &tt) {
  if (tt.has_value())
    return os << *tt;
  else
    return os << "<<<std::nullopt>>>";
}
