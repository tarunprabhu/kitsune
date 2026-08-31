//===- GPUMemUtils.cpp - Utilities for GPU memory -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Types and utilities to work with GPU memory.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/GPUMemUtils.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

template <> std::optional<GPUDynMemAllocKind> llvm::fromInt(int64_t v) {
  switch (v) {
  case 1: return GPUDynMemAllocKind::Global;
  case 2: return GPUDynMemAllocKind::UVM;
  default: return std::nullopt;
  }
}

template <> std::optional<GPUDynMemAllocKind> llvm::fromString(StringRef s) {
  return StringSwitch<std::optional<GPUDynMemAllocKind>>(s)
      .Case("global", GPUDynMemAllocKind::Global)
      .Case("uvm", GPUDynMemAllocKind::UVM)
      .Default(std::nullopt);
}

template <> std::string llvm::toString(const GPUDynMemAllocKind &memAllocKind) {
  switch (memAllocKind) {
  case GPUDynMemAllocKind::Global: return "global";
  case GPUDynMemAllocKind::UVM: return "uvm";
  }
  llvm_unreachable("toString: GPUDynMemAllocKind not handled!");
}
