//===- TTUIDtils.cpp - Utilities to deal with tapir target ids ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for TTID's.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/TTIDUtils.h"

using namespace llvm;

// The tapir targets that generate embedded bitcode.
static constexpr TTID ttbcs[] = {TTID::Cuda, TTID::Hip};

ArrayRef<TTID> llvm::ttsGenEmbBC() { return ttbcs; }

bool llvm::doesTTGenEmbBC(TTID tt) {
  for (TTID ttbc : ttsGenEmbBC())
    if (ttbc == tt)
      return true;
  return false;
}

bool llvm::isGPUTT(TTID tt) {
  switch (tt) {
  case TTID::Cuda:
  case TTID::Hip:
    return true;
  default:
    return false;
  }
}
