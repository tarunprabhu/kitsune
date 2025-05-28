//===- TapirTargetIDs.h - Tapir target ID's --------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file enumerates the available Tapir lowering targets and other types
// that are shared between the front and middle-ends.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_TAPIR_TAPIR_H
#define LLVM_FRONTEND_TAPIR_TAPIR_H

#include <cstdint>

namespace llvm {

/// An enumeration that may be set to a boolean value or unset.
enum class MaybeBool {
  Off, /// The value is set to false
  On,  /// The value is set to true
  Any  /// The value is unset
};

/// The identifiers for the known tapir targets.
enum class TapirTargetID : uint8_t {
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
};

/// The loop spawning strategy.
enum class TapirSpawnStrategy {
  Sequential,       /// Sequenial (no spawning)
  DivideAndConquer, /// Divide and conquer
  GPU               /// GPU-centric spawning strategy. Currently unused.
};

} // namespace llvm

#endif // LLVM_FRONTEND_TAPIR_TAPIR_H
