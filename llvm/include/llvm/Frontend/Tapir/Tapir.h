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

#include "llvm/ADT/StringRef.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"

namespace llvm {

/// An enumeration that may be set to a boolean value or unset.
enum class MaybeBool {
  Off, /// The value is set to false
  On,  /// The value is set to true
  Any  /// The value is unset
};

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

/// @{
/// Serialization functions for various types.

std::string toString(const TapirTargetID &);
std::string toString(const MaybeBool &);

raw_ostream &operator<<(raw_ostream &os, const TapirTargetID &);
raw_ostream &operator<<(raw_ostream &os, const TapirSpawnStrategy &);
raw_ostream &operator<<(raw_ostream &os, const OptimizationLevel &);
raw_ostream &operator<<(raw_ostream &os, const FPOpFusion::FPOpFusionMode &);
raw_ostream &operator<<(raw_ostream &os, const MaybeBool &);
/// @}

} // namespace llvm

#endif // LLVM_FRONTEND_TAPIR_TAPIR_H
