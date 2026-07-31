//===- FromString.cpp - Deserialization functions for Kitsune -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions to parse Kitsune-specific types from their string representations.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/FromString.h"
#include "kitsune/Core/Instrumentation.h"
#include "kitsune/Core/Tapir.h"
#include "kitsune/Support/MaybeBool.h"
#include "llvm/ADT/StringSwitch.h"

using namespace llvm;

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

template <> std::optional<TapirSpawnStrategy> llvm::fromString(StringRef s) {
  return StringSwitch<std::optional<TapirSpawnStrategy>>(s)
      .Case("seq", TapirSpawnStrategy::Sequential)
      .Case("dac", TapirSpawnStrategy::DivideAndConquer)
      .Case("gpu", TapirSpawnStrategy::GPU)
      .Case("basic", TapirSpawnStrategy::Basic)
      .Default(std::nullopt);
}

template <> std::optional<MaybeBool> llvm::fromString(StringRef s) {
  return StringSwitch<std::optional<MaybeBool>>(s)
      .Case("off", MaybeBool::Off)
      .Case("on", MaybeBool::On)
      .Case("any", MaybeBool::Any)
      .Default(std::nullopt);
}

template <> std::optional<InstrumentKind> llvm::fromString(StringRef s) {
  return StringSwitch<std::optional<InstrumentKind>>(s)
      .Case("generic", InstrumentKind::Generic)
      .Case("papi", InstrumentKind::PAPI)
      .Case("timer", InstrumentKind::Timer)
      .Default(std::nullopt);
}

template <> std::optional<InstrumentUnit> llvm::fromString(StringRef s) {
  return StringSwitch<std::optional<InstrumentUnit>>(s)
      .Case("loop", InstrumentUnit::Loop)
      .Case("thread", InstrumentUnit::Thread)
      .Default(std::nullopt);
}
