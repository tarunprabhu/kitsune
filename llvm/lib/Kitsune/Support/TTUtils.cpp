//===- TTUtils.cpp - Utilities to deal with tapir target ids --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to get "properties" of tapir targets.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/TTUtils.h"

using namespace llvm;

// ----------- THIS MUST BE UPDATED WHEN A NEW TAPIR TARGET IS ADDED -----------
//
// All known tapir targets.
// TODO: Update this when GPUABI is supported.
static constexpr TTID tts[] = {
    TTID::None,     TTID::Serial, TTID::Cuda,   TTID::Hip,
    TTID::OpenCilk, // TTID::GPUABI,
    TTID::Qthreads, TTID::Realm,  TTID::Lambda, TTID::OMPTask, TTID::OpenMP};

// The tapir targets that generate embedded bitcode.
static constexpr TTID ttbcs[] = {TTID::Cuda, TTID::Hip};

ArrayRef<TTID> llvm::ttsAll() { return tts; }

ArrayRef<TTID> llvm::ttsGenEmbBC() { return ttbcs; }

bool llvm::doesTTGenEmbBC(TTID tt) {
  for (TTID ttbc : ttsGenEmbBC())
    if (ttbc == tt)
      return true;
  return false;
}
