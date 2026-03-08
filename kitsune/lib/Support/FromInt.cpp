//===- FromInt.cpp - Conversions from ints --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions to convert integers to Kitsune-specific types.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/FromInt.h"
#include "kitsune/Core/Tapir.h"

using namespace llvm;

template <> std::optional<TTID> llvm::fromInt(int64_t v) {
  switch (v) {
  case 0x0:
    return TTID::Nolo;
  case 0x1:
    return TTID::Serial;
  case 0x2:
    return TTID::Cuda;
  case 0x4:
    return TTID::Hip;
  case 0x8:
    return TTID::OpenCilk;
  case 0x20:
    return TTID::Qthreads;
  case 0x40:
    return TTID::Realm;
  case 0x80:
    return TTID::Lambda;
  case 0x100:
    return TTID::OMPTask;
  case 0x200:
    return TTID::OpenMP;
  case 0x400:
    return TTID::Pthreads;
  case 0x800:
    return TTID::Custom;
  default:
    return std::nullopt;
  }
}

template <> std::optional<TapirSpawnStrategy> llvm::fromInt(int64_t v) {
  switch (v) {
  case 0x1:
    return TapirSpawnStrategy::Sequential;
  case 0x2:
    return TapirSpawnStrategy::DivideAndConquer;
  case 0x3:
    return TapirSpawnStrategy::GPU;
  case 0x4:
    return TapirSpawnStrategy::Basic;
  default:
    return std::nullopt;
  }
}
