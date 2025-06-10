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

/// The identifiers for the known tapir targets.
///
/// These are some useful constraints that it would be useful to maintain.
/// The "None" tapir target does not perform any lowering - instead, the tapir
/// instructions are retained in the tapir loop even after it has been
/// "lowered". This should always have an integer value of 0. The serial tapir
/// target simply lowers the tapir loop to a serial loop. This should have an
/// integer value of 1 because it "makes sense" in this context - a serial loop
/// can be thought of as a special case of a parallel loop where only a single
/// iteration is executing at a time.
///
/// The integer values of some of the tapir targets are hardcoded because these
/// are used in the tests. Inserting a new tapir target to keep this enum in
/// alphabetical order would entail changing a number of tests. When adding a
/// new tapir target, it is ok if the values are not in ascending order as long
/// as the names of the targets are in ascending order.
enum class TapirTargetID : uint8_t {
  None = 0,   // Do not lower
  Serial = 1, // Lower to serial projection
  Cuda = 2,   // Lower to Kitsune's NVIDIA GPU runtime (cuda)
  Hip = 3,    // Lower to Kitsune's AMD GPU runtime (hip)
  Lambda,     // Lower to generic Lambda ABI
  OMPTask,    // Lower to OpenMP task ABI
  OpenCilk,   // Lower to OpenCilk ABI
  OpenMP,     // Lower to OpenMP (TODO: Needs to be updated)
  Qthreads,   // Lower to Qthreads (TODO: Needs to be updated)
  Realm,      // Lower to Realm (TODO: Needs to be updated)
};

/// An enumeration that may be set to a boolean value or unset.
enum class MaybeBool {
  Off, /// The value is set to false
  On,  /// The value is set to true
  Any  /// The value is unset
};

/// The loop spawning strategy.
enum class TapirSpawnStrategy {
  Sequential,       /// Sequenial (no spawning)
  DivideAndConquer, /// Divide and conquer
  GPU               /// GPU-centric spawning strategy. Currently unused.
};

} // namespace llvm

#endif // LLVM_FRONTEND_TAPIR_TAPIR_H
