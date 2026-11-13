//===- ReductionStrategy.cpp - The reduction strategy enum ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The core reduction strategy enum.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ReductionStrategy.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/StringSwitch.h"

using namespace llvm;

template <> StringRef llvm::toString<ReductionStrategy>() {
  return "llvm::ReductionStrategy";
}

template <> std::string llvm::toString(const ReductionStrategy &strategy) {
  switch (strategy) {
  case ReductionStrategy::Auto: return "auto";
  case ReductionStrategy::Naive: return "naive";
  case ReductionStrategy::Direct: return "direct";
  case ReductionStrategy::GPUSharedMemory: return "shmem";
  case ReductionStrategy::GPUWarpShuffle: return "wshfl";
  case ReductionStrategy::GPUWarpShuffleWithSharedMemory: return "wsfhlshmem";
  }
  llvm_unreachable("toString: ReductionStrategy not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const ReductionStrategy &strategy) {
  os << toString(strategy);
  return os;
}

template <> std::optional<ReductionStrategy> llvm::fromString(StringRef s) {
  return StringSwitch<std::optional<ReductionStrategy>>(s)
      .Case("auto", ReductionStrategy::Auto)
      .Case("naive", ReductionStrategy::Naive)
      .Case("direct", ReductionStrategy::Direct)
      .Case("shmem", ReductionStrategy::GPUSharedMemory)
      .Case("wshfl", ReductionStrategy::GPUWarpShuffle)
      .Case("wshflshmem", ReductionStrategy::GPUWarpShuffleWithSharedMemory)
      .Default(std::nullopt);
}

template <> std::optional<ReductionStrategy> llvm::fromInt(int64_t v) {
  switch (v) {
  case 0x0: return ReductionStrategy::Auto;
  case 0x1: return ReductionStrategy::Naive;
  case 0x2: return ReductionStrategy::Direct;
  case 0x6: return ReductionStrategy::GPUSharedMemory;
  case 0xA: return ReductionStrategy::GPUWarpShuffle;
  case 0xE: return ReductionStrategy::GPUWarpShuffleWithSharedMemory;
  default: return std::nullopt;
  }
}

bool llvm::isCPUStrategy(ReductionStrategy strategy) {
  switch (strategy) {
  case ReductionStrategy::Auto:
  case ReductionStrategy::Naive:
  case ReductionStrategy::Direct: //
    return true;
  case ReductionStrategy::GPUSharedMemory:
  case ReductionStrategy::GPUWarpShuffle:
  case ReductionStrategy::GPUWarpShuffleWithSharedMemory: //
    return false;
  }
  llvm_unreachable("isCPUStrategy: ReductionStrategy not handled");
}

bool llvm::isGPUStrategy(ReductionStrategy strategy) {
  switch (strategy) {
  case ReductionStrategy::Auto:
  case ReductionStrategy::Naive:
  case ReductionStrategy::Direct:
  case ReductionStrategy::GPUSharedMemory:
  case ReductionStrategy::GPUWarpShuffle:
  case ReductionStrategy::GPUWarpShuffleWithSharedMemory: //
    return true;
  }
  llvm_unreachable("isGPUStrategy: ReductionStrategy not handled");
}
