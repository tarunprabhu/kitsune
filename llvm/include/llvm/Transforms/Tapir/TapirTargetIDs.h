//===- TapirTargetIDs.h - Tapir target ID's --------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file enumerates the available Tapir lowering targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TAPIR_TARGET_IDS_H
#define LLVM_TAPIR_TARGET_IDS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"

namespace llvm {

enum class TapirTargetID {
  None = 1, // Perform no lowering
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

/// Parse the Tapir target from a string. If the string is not a valid tapir
/// target, return std::nullopt.
std::optional<TapirTargetID> parseTapirTarget(StringRef s);

// Serialize the Tapir target into the given output stream. This will write a
// string representation that is compatible with the -ftapir argument used in
// clang.
raw_ostream &operator<<(raw_ostream &os, const TapirTargetID &Target);

/// Virtual base class for Target-specific options.
class TapirTargetOptions {
public:
  enum TapirTargetOptionsKind {
    TTO_None,
    TTO_Serial,
    TTO_Cuda,
    TTO_Hip,
    TTO_Lambda,
    TTO_OMPTask,
    TTO_OpenCilk,
    TTO_OpenMP,
    TTO_Qthreads,
    TTO_Realm
  };

private:
  const TapirTargetOptionsKind Kind;

protected:
  TapirTargetOptions(TapirTargetOptionsKind K) : Kind(K) {}

public:
  TapirTargetOptions(const TapirTargetOptions &) = delete;
  TapirTargetOptions &operator=(const TapirTargetOptions &) = delete;
  virtual ~TapirTargetOptions() = default;

  TapirTargetOptionsKind getKind() const { return Kind; }

  /// Top-level method for cloning TapirTargetOptions.  Defined in
  /// TargetLibraryInfo.
  virtual TapirTargetOptions *clone() const = 0;
};

} // end namespace llvm

#endif // LLVM_TAPIR_TARGET_IDS_H
