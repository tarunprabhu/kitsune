//===- TapirTargetIDs.h - Tapir target ID's --------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file enumerates the available Tapir lowering targets and other types
// that are shared between the frontend and middle-ends.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TAPIR_TARGET_IDS_H
#define LLVM_TAPIR_TARGET_IDS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// The identifiers for the known tapir targets.
enum class TapirTargetID {
  None = 0, // Perform no lowering
  Serial,   // Lower to serial projection
  Cuda,     // Lower to Cuda ABI
  Hip,      // Lower to the Hip (AMD GPU) ABI
  Lambda,   // Lower to generic Lambda ABI
  OMPTask,  // Lower to OpenMP task ABI
  OpenCilk, // Lower to OpenCilk ABI
  OpenMP,   // Lower to OpenMP (TODO: Needs to be updated)
  Qthreads, // Lower to Qthreads (TODO: Needs to be updated)
  Realm,    // Lower to Realm (TODO: Needs to be updated)
  Last_TapirTargetID
};

/// The loop spawning strategy.
enum class TapirSpawnStrategy {
  Sequential,       /// Sequenial (no spawning)
  DivideAndConquer, /// Divide and conquer
  GPU               /// GPU-centric spawning strategy. Currently unused.
};

// Serialize the Tapir target into the given output stream. This will write a
// string representation that is compatible with the -ftapir argument used in
// clang.
raw_ostream &operator<<(raw_ostream &os, const TapirTargetID &Target);

/// Serialization functions to help with debugging and more useful verbose mode
/// output.
raw_ostream &operator<<(raw_ostream &os, const TapirSpawnStrategy &strategy);

} // namespace llvm

#endif // LLVM_TAPIR_TARGET_IDS_H
