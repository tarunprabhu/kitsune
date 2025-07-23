//===- TTUtils.h - Utilities to deal with tapir target ids -----*- C++ -*--===//
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

#ifndef KITSUNE_SUPPORT_TTUTILS_H
#define KITSUNE_SUPPORT_TTUTILS_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/ArrayRef.h"

namespace llvm {

// TODO: This list must be updated if a tapir target that generates embedded
// bitcode is added.
/// The tapir targets that generate embedded bitcode.
constexpr TTID ttsUsingEmbBC[] = {TTID::Cuda, TTID::Hip};

// FIXME: It may be possible to use cmake to populate this array. Otherwise,
// the code in llvm/CMakeLists.txt must be kept in sync with this.
// FIXME: Update this when GPUABI is supported.
// TODO: This list must be updated when a tapir target is added/removed.
/// All known tapir targets.
constexpr TTID ttsAll[] = {TTID::Nolo,     TTID::Serial,   TTID::Cuda,
                           TTID::Hip,      TTID::OpenCilk, /* TTID::GPUABI, */
                           TTID::Qthreads, TTID::Realm,    TTID::Lambda,
                           TTID::OMPTask,  TTID::OpenMP};

/// Check if the given tapir target generates embedded bitcode.
inline bool ttUsesEmbBC(TTID tt) {
  for (TTID ttid : ttsUsingEmbBC)
    if (ttid == tt)
      return true;
  return false;
}

} // namespace llvm

#endif // KITSUNE_SUPPORT_TTUTILS_H
