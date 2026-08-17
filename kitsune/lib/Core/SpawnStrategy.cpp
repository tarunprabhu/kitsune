//===- SpawnStrategy.cpp - The spawn strategsy enum -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define the core spawn strategy enum.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/SpawnStrategy.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"

using namespace llvm;

template <> StringRef llvm::toString<TapirSpawnStrategy>() {
  return "llvm::TapirSpawnStrategy";
}

template <> std::string llvm::toString(const TapirSpawnStrategy &strategy) {
  switch (strategy) {
  case TapirSpawnStrategy::Sequential: return "Sequential";
  case TapirSpawnStrategy::DivideAndConquer: return "Divide and conquer";
  case TapirSpawnStrategy::GPU: return "GPU";
  case TapirSpawnStrategy::Basic: return "Basic";
  }
  llvm_unreachable("operator<<: TapirSpawnStrategy not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const TapirSpawnStrategy &strategy) {
  os << toString(strategy);
  return os;
}

template <> std::optional<TapirSpawnStrategy> llvm::fromString(StringRef s) {
  return StringSwitch<std::optional<TapirSpawnStrategy>>(s)
      .Case("seq", TapirSpawnStrategy::Sequential)
      .Case("dac", TapirSpawnStrategy::DivideAndConquer)
      .Case("gpu", TapirSpawnStrategy::GPU)
      .Case("basic", TapirSpawnStrategy::Basic)
      .Default(std::nullopt);
}

template <> std::optional<TapirSpawnStrategy> llvm::fromInt(int64_t v) {
  switch (v) {
  case 0x1: return TapirSpawnStrategy::Sequential;
  case 0x2: return TapirSpawnStrategy::DivideAndConquer;
  case 0x3: return TapirSpawnStrategy::GPU;
  case 0x4: return TapirSpawnStrategy::Basic;
  default: return std::nullopt;
  }
}
