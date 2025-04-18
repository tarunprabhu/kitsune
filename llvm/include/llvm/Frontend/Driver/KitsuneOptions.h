//===--- KitsuneOptions.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Kitsune-specific frontend options common to clang and
// flang. These are not exactly the same as TapirTargetOptions which are the
// options used by the various tapir targets. These options can affect parsing
// and lowering of the various languages supported by Kitsune.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_DRIVER_KITSUNE_OPTIONS_H
#define LLVM_FRONTEND_DRIVER_KITSUNE_OPTIONS_H

#include "kitsune/Config/config.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/Tapir/Tapir.h"

#include <vector>

namespace llvm::driver {

/// Options that are Kitsune-specific. These affect both the Kitsune "language"
/// i.e. forall, spawn, sync etc. and the backend code-generation via Tapir.
class KitsuneOptions {
private:
  /// Is a Kitsune frontend being used. The frontend could be used without a
  /// tapir target, so we can't use the @ref TapirTarget field to determine
  /// whether we are using Kitsune.
  unsigned kitsuneFrontend : 1;

  /// Is "Kokkos mode" enabled.
  unsigned kokkos : 1;

  /// If "Kokkos mode" is enabled, should the initialization of libkokkoscore
  /// be overrident.
  unsigned kokkosNoInit : 1;

  /// Should loop strip-mining be enabled.
  unsigned stripmineLoops : 1;

  /// Enable verbose mode for the tapir target LLVM passes. This is different
  /// from -mllvm -debug-only=<tapir-target-llvm-pass-name>. When the verbose
  /// flag is set, the passes may print some subset of the information that is
  /// printed in full debug mode. However, there is no guarantee that any given
  /// tapir target LLVM pass will print anything at all.
  unsigned tapirVerbose : 1;

  /// If true, code will be generated to enable verbose mode in the Kitsune
  /// runtime. This obviates the need to set an environment variable to turn on
  /// verbose mode when the compiled code is run. The effect of this will only
  /// be visible in certain tapir targets because not all use Kitsune's runtime.
  unsigned kitrtVerbose : 1;

  /// The "primary" tapir target for code generation. The "inline" tapir
  /// targets that are attached to specific constructs are separate from this.
  /// This is set to the value of the the --tapir option passed on the command
  /// line. It is option because we do not have a default tapir target, even
  /// when using the Kitsune frontends (kitcc, kitfc etc.)
  std::optional<llvm::TapirTargetID> tapirTarget = std::nullopt;

  /// If this is non-zero, the number of threads per block to use.
  unsigned fixedThreadsPerBlock = 0;

  /// If this is non-zero, the maximum number of threads per block to use. This
  /// may be used in conjunction with @ref threadsPerBlock, in which case this
  /// value must be greater than or equal to @ref threadsPerBlock.
  unsigned maxThreadsPerBlock = 0;

  /// The path to LLD that was built with Kitsune. If clang is invoked from the
  /// build directory, this will be the lld that is in the build directory.
  std::string lld = "";

  /// The NVIDIA GPU architecture for which to generate code. This is only
  /// relevant for the cuda tapir target, although the default is always set.
  /// This is a string and not an enum because it is not clear if anything is to
  /// be gained by making it an enum. So far, all uses of this are as a string.
  std::string cudaArch = KITSUNE_CUDA_ARCH_DEFAULT;

  /// The virtual architecture (compute_*), for the @ref cudaArch. This will
  /// be computed by the driver from either the default value of @ref cudaArch,
  /// or from the --tapir-cuda-arch option if it was provided.
  std::string cudaVirtArch = "";

  /// The PTX version for the @ref cudaArch. This will be computed by the driver
  /// from either the default value of @ref cudaArch or from
  /// the --tapir-cuda-arch option if it was provided.
  std::string cudaPTXVersion = "";

  /// NVIDIA GPU target features, computed by the driver, for @ref cudaArch.
  /// This is a string that can be used by the NVPTX module that is generated
  /// when lowering with the cuda tapir target.
  std::string cudaFeatures = "";

  /// Absolute path to the cuda runtime bitcode file. This will only be
  /// non-empty when the cuda tapir target is enabled.
  std::string cudaRuntimeBCFile = "";

  /// The AMD GPU architecture for which to generate code. This is only
  /// relevant for the hip tapir target, although the default is always set.
  /// This is a string and not an enum because it is not clear if anything
  /// is to be gained by making it an enum. So far, all uses of this are as
  /// a string.
  std::string hipArch = KITSUNE_HIP_ARCH_DEFAULT;

  /// AMD GPU target features, computed by the driver, for @ref hipArch. This is
  /// a string that can be used by the NVPTX module that is generated when
  /// lowering with the hip tapir target.
  std::string hipFeatures = "";

  /// Absolute paths to the bitcode files that are to be provided to the AMDGPU
  /// backend when the hip tapir target is enabled. The contents of the list are
  /// computed by the driver based on @ref hipArch and any other target-specific
  /// options that were provided.
  std::vector<std::string> hipRuntimeBCFiles;

  /// The value of the sramecc feature. Setting this to true implies
  /// sramecc+, false implies sramecc-. Leaving this at std::nullopt implies
  /// that the sramecc feature is unspecified.
  ///
  /// FIXME: This should default to Maybe::Any. This was initially
  /// implemented in LLVM 20 as a replacement for the --hipabi-xnack LLVM
  /// option which defaulted to true. This is intended to be set to 'on'
  /// during a transition period.
  MaybeBool hipSRAMECC = MaybeBool::On;

  /// The value of the xnack feature. Setting this to true implies xnack+, false
  /// implies xnack-. Leaving this at std::nullopt implies that the xnack
  /// feature is unspecified.
  ///
  /// FIXME: This should default to std::nullopt. This was initially implemented
  /// in LLVM 20 as a replacement for the --hipabi-xnack LLVM option which
  /// defaulted to true. This is intended to be set to 'on' during a transition
  /// period.
  MaybeBool hipXnack = MaybeBool::On;

  /// The hip bitcode files needed by the hip tapir target. This will be
  /// computed by the driver from either the default value of @ref cudaArch, or
  /// from the --tapir-hip-arch option if it was provided.
  std::vector<std::string> hipBitcodeFiles;

  /// Path to the OpenCilk ABI bitcode file. This will only be non-empty if the
  /// OpenCilk tapir target is enabled.
  std::string openCilkRuntimeBCFile = "";

public:
  void setKitsuneFrontend(bool kitsuneFrontend = true) {
    this->kitsuneFrontend = kitsuneFrontend;
  }

  void setKokkos(bool kokkos = true) { this->kokkos = kokkos; }

  void setKokkosNoInit(bool kokkosNoInit = true) {
    this->kokkosNoInit = kokkosNoInit;
  }

  void setStripmineLoops(bool stripmineLoops = true) {
    this->stripmineLoops = stripmineLoops;
  }

  void setKitrtVerbose(bool verbose = true) {
    this->kitrtVerbose = verbose;
  }

  void setTapirVerbose(bool verbose = true) {
    this->tapirVerbose = verbose;
  }

  void setTapirTarget(llvm::TapirTargetID tapirTarget) {
    this->tapirTarget = tapirTarget;
  }

  void setOpenCilkRuntimeBCFile(llvm::StringRef path) {
    this->openCilkRuntimeBCFile = path;
  }

  void setFixedThreadsPerBlock(unsigned threadsPerBlock) {
    this->fixedThreadsPerBlock = threadsPerBlock;
  }

  void setMaxThreadsPerBlock(unsigned threadsPerBlock) {
    this->maxThreadsPerBlock = threadsPerBlock;
  }

  void setCudaArch(llvm::StringRef arch) { this->cudaArch = arch; }

  void setCudaVirtArch(llvm::StringRef arch) { this->cudaVirtArch = arch; }

  void setCudaPTXVersion(llvm::StringRef ptx) { this->cudaPTXVersion = ptx; }

  void setCudaFeatures(llvm::StringRef features) {
    this->cudaFeatures = features;
  }

  void setHipArch(llvm::StringRef arch) { this->hipArch = arch; }

  void setHipFeatures(llvm::StringRef features) {
    this->hipFeatures = features;
  }

  void setHipSramECC(MaybeBool ecc) { this->hipSRAMECC = ecc; }

  void setHipXnack(MaybeBool xnack) { this->hipXnack = xnack; }

  void addHipRuntimeBCFile(llvm::StringRef file) {
    this->hipRuntimeBCFiles.push_back(file.str());
  }

  void setLLD(llvm::StringRef lld) { this->lld = lld; }

  bool isKitsuneFrontend() const { return kitsuneFrontend; }

  bool hasTapirTarget() const { return tapirTarget.has_value(); }

  bool getKokkos() const { return kokkos; }

  bool getKokkosNoInit() const { return kokkosNoInit; }

  bool getStripmineLoops() const { return stripmineLoops; }

  bool getTapirVerbose() const { return tapirVerbose; }

  bool getKitrtVerbose() const { return kitrtVerbose; }

  std::optional<llvm::TapirTargetID> getTapirTarget() const {
    return tapirTarget;
  }

  unsigned getFixedThreadsPerBlock() const { return fixedThreadsPerBlock; }

  unsigned getMaxThreadsPerBlock() const { return maxThreadsPerBlock; }

  llvm::StringRef getLLD() const { return lld; }

  llvm::StringRef getCudaArch() const { return cudaArch; }

  llvm::StringRef getCudaVirtArch() const { return cudaVirtArch; }

  llvm::StringRef getCudaPTXVersion() const { return cudaPTXVersion; }

  llvm::StringRef getCudaFeatures() const { return cudaFeatures; }

  llvm::StringRef getCudaRuntimeBCFile() const { return cudaRuntimeBCFile; }

  llvm::StringRef getHipArch() const { return hipArch; }

  llvm::StringRef getHipFeatures() const { return hipFeatures; }

  MaybeBool getHipSRAMECC() const { return hipSRAMECC; }

  MaybeBool getHipXnack() const { return hipXnack; }

  const std::vector<std::string> &getHipRuntimeBCFiles() const {
    return hipRuntimeBCFiles;
  }

  llvm::StringRef getOpenCilkRuntimeBCFile() const {
    return openCilkRuntimeBCFile;
  }
};

} // namespace llvm::driver

#endif // LLVM_FRONTEND_DRIVER_KITSUNE_OPTIONS_H
