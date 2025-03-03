//===- TapirTargetOptions.h - Tapir target options objects -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the base options object for all Tapir targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TAPIR_TARGET_OPTIONS_H
#define LLVM_TAPIR_TARGET_OPTIONS_H

#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

namespace llvm {

/// Virtual base class for Target-specific options.
class TapirTargetOptions {
public:
  enum TapirTargetOptionsKind {
    TTO_None = 1,
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
  const TapirTargetOptionsKind kind;

protected:
  /// Enable verbose mode on the tapir target. Not all tapir targets implement
  /// verbose mode, so there is no guarantee that this has any effect on any
  /// given tapir target.
  unsigned verbose : 1;

  /// If true, set the Kitsune runtime in verbose mode. Not all tapir targets
  /// use Kitsune's runtime. In such cases, setting this to true will no effect.
  unsigned runtimeVerbose : 1;

protected:
  explicit TapirTargetOptions(TapirTargetOptionsKind kind) : kind(kind) {}

public:
  explicit TapirTargetOptions(const TapirTargetOptions &) = default;
  virtual ~TapirTargetOptions() = default;

  TapirTargetOptions &operator=(const TapirTargetOptions &) = delete;

  void setVerbose(bool verbose = true) { this->verbose = verbose; }
  void setRuntimeVerbose(bool verbose = true) {
    this->runtimeVerbose = verbose;
  }

  TapirTargetOptionsKind getKind() const { return kind; }
  bool getVerbose() const { return verbose; }
  bool getRuntimeVerbose() const { return runtimeVerbose; }

  /// Set the options from the command line options. Each tapir target specifies
  /// its own set of command line options, though some are shared between the
  /// tapir targets. This is only intended to be used when setting up the
  /// options object in LLVM. In practice, this will only happen when driving
  /// the tapir lowering pipeline using opt. Frontends like clang and flang
  /// should never call this, but should set up the object directly.
  ///
  /// Subclasses of this class are responsible for calling the parent's
  /// implementation of this method to ensure that all command line options are
  /// correctly setup.
  ///
  virtual void readClOptions();

  /// Clone this options object.
  virtual TapirTargetOptions *clone() const = 0;
};

/// Serialization functions to help in debugging/verbose mode.
raw_ostream &operator<<(raw_ostream &os, const bool &);

} // namespace llvm

#endif // LLVM_TAPIR_TARGET_OPTIONS_H
