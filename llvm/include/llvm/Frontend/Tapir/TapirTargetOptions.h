//===- TapirTargetOptions.h - Tapir target options objects -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The options objects for all the known tapir targets as well as a common
// options "wrapper" object.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TAPIR_TARGET_OPTIONS_H
#define LLVM_TAPIR_TARGET_OPTIONS_H

#include "llvm/Frontend/Tapir/Tapir.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Target/TargetOptions.h"

#include <set>

namespace llvm {

namespace driver {
class KitsuneOptions;
}

using KitsuneOptions = driver::KitsuneOptions;
using FPOpFusionMode = FPOpFusion::FPOpFusionMode;

/// Options for the tapir targets. The same options object is used for all
/// tapir targets. We could split it into types for each tapir target, but it
/// doesn't really help with anything. Having everything "under one roof" makes
/// it a shade easier to support multiple tapir targets in a single compilation.
class TapirTargetOptions {
private:
  /// The primary tapir target. This is guaranteed to always be set since there
  /// must always be at least one tapir target.
  TapirTargetID tt;

  /// When multiple tapir targets are supported, these are the secondary tapir
  /// targets. This set will *not* include the primary tapir target.
  ///
  /// As of April 2025, there is preliminary support for multiple tapir targets.
  /// In this scheme, the primary tapir target is the one specified with the
  /// --tapir= option on the frontend. The secondary tapir targets are those
  /// specified on individual syntactic constructs (such as forall loops and
  /// spawn statements).
  std::set<TapirTargetID> tts;

  /// Options common to all tapir targets
  /// @{
  /// Enable verbose mode on the tapir target. Not all tapir targets implement
  /// verbose mode, so there is no guarantee that this has any effect on a
  /// given tapir target.
  unsigned tapirVerbose : 1;

  /// If true, set the Kitsune runtime in verbose mode. Not all tapir targets
  /// use Kitsune's runtime. In such cases, setting this to true will no effect.
  unsigned kitrtVerbose : 1;
  /// @}

  /// The optimization level set on the command line. This level will be used
  /// for both the middle-end optimizations on the kernel functions and the
  /// backend GPU code generators (including external assemblers as needed).
  OptimizationLevel optLevel;

  /// How to use fuse floating-point operations.
  FPOpFusion::FPOpFusionMode fpOpFusionMode = FPOpFusion::Strict;

  /// Options common to the GPU tapir targets
  /// @{
  /// If this is non-zero, this value will be used when launching all kernels
  /// which do not already have a custom FixedThreadsPerBlock value. If this is
  /// not set, the number of threads per block to use will be determined by the
  /// runtime.
  unsigned fixedThreadsPerBlock = 0;

  /// If this is non-zero, the threads per block will not be allowed to exceed
  /// this value.
  unsigned maxThreadsPerBlock = 0;

  /// The path to LLD that was built with Kitsune. If clang is invoked from the
  /// build directory, this will be the lld that is in the build directory.
  std::string lld;
  /// @}

  /// Options for the cuda tapir target
  /// @{
  /// The NVIDIA GPU architecture for which to generate code.
  std::string cudaArch;

  /// The virtual architecture (compute_*), for the @ref cudaArch.
  std::string cudaVirtArch;

  /// The PTX version for the @ref cudaArch.
  std::string cudaPTXVersion;

  /// Cuda target features for the current @ref cudaArch. This is a string with
  /// the format: +feature1,+feature2,-feature3...
  std::string cudaFeatures;

  /// The absolute path to the cuda runtime bitcode file.
  std::string cudaRuntimeBCFile;
  /// @}

  /// Options for the hip tapir target
  /// @{
  /// The AMD GPU architecture for which to generate code.
  std::string hipArch;

  /// The value of the sramecc feature.
  MaybeBool hipSRAMECC;

  /// The value of the xnack feature.
  MaybeBool hipXnack;

  /// The target features for the current hip GPU architecture. This is a string
  /// with the format: '+feature1,+feature2,-feature3...'.
  std::string hipFeatures;

  /// The hip bitcode files needed by the hip tapir target. This will be
  /// computed by the driver from either the default value of @ref cudaArch, or
  /// from the --tapir-hip-arch option if it was provided.
  std::vector<std::string> hipRuntimeBCFiles;
  /// @}

  /// Options for the OpenCilk tapir target
  /// @{
  std::string openCilkRuntimeBCFile;
  /// @}

private:
  TapirTargetOptions() = default;

  /// Create an options object with the given primary tapir target.
  TapirTargetOptions(TapirTargetID tt);

  /// Create an options object with the given primary tapir target and the given
  /// secondary tapir targets.
  TapirTargetOptions(TapirTargetID tt, const std::vector<TapirTargetID> tts);

public:

  /// If a tapir target is set, does the tapir target support lowering.
  ///
  /// The "none" "pseudo tapir target" is useful to check that the frontend
  /// correctly generates tapir instructions and that these make their way
  /// through the middle-end optimizations. But they are never lowered.
  bool lower() const;

  /// Create a clone of this options object.
  std::unique_ptr<TapirTargetOptions> clone() const;

  TapirTargetID getTapirTargetID() const { return tt; }

  bool getTapirVerbose() const { return tapirVerbose; }
  bool getKitrtVerbose() const { return kitrtVerbose; }
  OptimizationLevel getOptLevel() const { return optLevel; }
  FPOpFusion::FPOpFusionMode getFPOpFusionMode() const {
    return fpOpFusionMode;
  }
  unsigned getFixedThreadsPerBlock() const { return fixedThreadsPerBlock; }
  unsigned getMaxThreadsPerBlock() const { return maxThreadsPerBlock; }
  StringRef getCudaArch() const { return cudaArch; }
  StringRef getCudaVirtArch() const { return cudaVirtArch; }
  StringRef getCudaPTXVersion() const { return cudaPTXVersion; }
  StringRef getCudaFeatures() const { return cudaFeatures; }
  StringRef getCudaRuntimeBCFile() const { return cudaRuntimeBCFile; }
  StringRef getHipArch() const { return hipArch; }
  MaybeBool getHipSramECC() const { return hipSRAMECC; }
  MaybeBool getHipXnack() const { return hipXnack; }
  StringRef getHipFeatures() const { return hipFeatures; }
  const std::vector<std::string> &getHipRuntimeBCFiles() const {
    return hipRuntimeBCFiles;
  }
  StringRef getLLD() const { return lld; }
  StringRef getOpenCilkRuntimeBCFile() const { return openCilkRuntimeBCFile; }

  /// Construct an options object from the given frontend options. If a tapir
  /// target ID is not set in the kitsune options, std::nullopt is returned.
  static std::optional<TapirTargetOptions>
  create(const KitsuneOptions &kitOpts, OptimizationLevel optLevel,
         FPOpFusionMode fpOpFusionMode);

  /// Construct an options object initialized from the command line options
  /// if the --tapir option was provided. Otherwise, return std::nullopt.
  static std::optional<TapirTargetOptions> createFromCommandLineOptions();
};

} // namespace llvm

#endif // LLVM_TAPIR_TARGET_OPTIONS_H
