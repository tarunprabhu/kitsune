//===- KitOptions.h - Kitsune options shared by all frontends ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Options shared by Kitsune's frontends. These options can affect parsing and
// lowering of the various languages supported by Kitsune.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_KIT_OPTIONS_H
#define KITSUNE_CORE_KIT_OPTIONS_H

#include "kitsune/Config/Config.h"
#include "kitsune/Core/Tapir.h"
#include "kitsune/Support/MaybeBool.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

namespace driver {

/// \addtogroup kitsune
/// @{

/// Kitsune-specific options. These affect both the Kitsune language extensions,
/// such as forall, spawn, sync etc., the middle-end optimizations, including
/// the lowering of parallel constructs by Tapir, and machine-code generation,
/// especially if something Kitsune-specific must be performed.
///
/// This is in the llvm::driver namespace because that is where objects that are
/// shared between LLVM frontends are added by convention. This object is
/// currently shared between clang and flang.
///
/// This is intended to be a POD (Plain Old Data) type. All accessor methods are
/// defined in this header, both for performance and to allow it to be used
/// without requiring the libLLVMKitCore to be linked into the user.
class KitOptions {
public:
  /// Should the loop stripmining pass be enabled by default.
  static constexpr bool defaultStripmineLoops = false;

  /// Is prefetching of managed memory between host and GPU enabled by default.
  static constexpr bool defaultGPUPrefetch = true;

  /// The default optimization level to assume. While Tapir is only useful with
  /// a non-zero optimization level, we set the default to 0 anyway to be
  /// consistent with the rest of the compiler and to require optimizations to
  /// be enabled by the user. Using a non-zero default complicates testing which
  /// is enough of a pain already.
  static constexpr unsigned defaultSpeedupLevel = 0;

  /// The default NVIDIA GPU architecture for which to generate code. This is
  /// only used if a NVIDIA GPU was not found on the system when using the cuda
  /// tapir target.
  static constexpr StringRef defaultCudaArch = kitCudaDefaultArch();

  /// The default AMD GPU architecture for which to generate code. This is
  /// only used if an AMD GPU was not found on the system when using the hip
  /// tapir target.
  static constexpr StringRef defaultHipArch = kitHipDefaultArch();

  /// The default value of SRAMECC for hip tapir targets.
  ///
  /// FIXME: This should probably default to MaybeBool::Any. This was initially
  /// implemented in LLVM 20 as a replacement for the --hipabi-sramecc LLVM
  /// option which defaulted to true. The default is set to on for a transition
  /// period while we determine which of the two - On and Any - is better for
  /// our use case.
  static constexpr MaybeBool defaultHipSRAMECC = MaybeBool::On;

  /// The default value of Xnack for hip tapir targets.
  ///
  /// FIXME: This should default to MaybeBool::Any. This was initially
  /// implemented in LLVM 20 as a replacement for the --hipabi-xnack LLVM option
  /// which defaulted to true. The default is set to on for a transition
  /// period while we determine which of the two - On and Any - is better for
  /// our use case.
  static constexpr MaybeBool defaultHipXnack = MaybeBool::On;

private:
  /// Is a Kitsune frontend being used. The frontend could be used without a
  /// tapir target, so we can't use the @ref tt field to determine whether we
  /// are using Kitsune.
  unsigned kitsuneFrontend : 1;

  /// Is "Kokkos mode" enabled.
  unsigned kokkos : 1;

  /// If "Kokkos mode" is enabled, should the initialization of libkokkoscore
  /// be overridden.
  unsigned kokkosNoInit : 1;

  /// Should loop strip-mining be enabled.
  unsigned stripmineLoops : 1;

  /// Enable prefetching managed memory with GPU tapir targets. If this is set
  /// to true, prefetch calls between host and device may be inserted. There is
  /// no guarantee because a profitability analysis may determine that there is
  /// no benefit to doing so.
  unsigned gpuPrefetch : 1;

  /// The "primary" tapir target for code generation. The "inline" tapir
  /// targets that are attached to specific constructs are separate from this.
  /// This is set to the value of the the --tapir option passed on the command
  /// line. It is optional because we may not have a default tapir target
  std::optional<llvm::TTID> tt = llvm::defaultTapirTarget;

  /// If this is not std::nullopt, it must be the path to a dynamic shared
  /// object that can be loaded as a tapir target plugin.
  std::string ttPlugin;

  /// If this is non-zero, the number of threads per block to use.
  unsigned fixedThreadsPerBlock = 0;

  /// If this is non-zero, the maximum number of threads per block to use. This
  /// may be used in conjunction with @ref threadsPerBlock, in which case this
  /// value must be greater than or equal to @ref threadsPerBlock.
  unsigned maxThreadsPerBlock = 0;

  /// The path to LLD that was built with Kitsune. This path is determined by
  /// the location of clang's executable. This allows it to be used from both
  /// the build and install directories. The logic for doing so was already in
  /// the driver and we just use it.
  std::string lld;

  /// The NVIDIA GPU architecture for which to generate code. This is only
  /// relevant for the cuda tapir target, although the default is always set.
  /// This is a string and not an enum because it is not clear if anything is to
  /// be gained by making it an enum. So far, all uses of this are as a string.
  std::string cudaArch = defaultCudaArch.str();

  /// The virtual architecture (compute_*), for the @ref cudaArch. This will
  /// be computed by the driver from either the default value of @ref cudaArch,
  /// or from the --tapir-cuda-arch option if it was provided.
  std::string cudaVirtArch;

  /// NVIDIA GPU target features, computed by the driver, for @ref cudaArch.
  /// This is a string that can be used by the NVPTX module that is generated
  /// when lowering with the cuda tapir target.
  std::string cudaFeatures;

  /// Absolute path to the cuda runtime bitcode file. This will only be
  /// non-empty when the cuda tapir target is enabled.
  std::string cudaRuntimeBCFile;

  /// The AMD GPU architecture for which to generate code. This is only
  /// relevant for the hip tapir target, although the default is always set.
  /// This is a string and not an enum because it is not clear if anything
  /// is to be gained by making it an enum. So far, all uses of this are as
  /// a string.
  std::string hipArch = defaultHipArch.str();

  /// AMD GPU target features, computed by the driver, for @ref hipArch. This is
  /// a string that can be used by the NVPTX module that is generated when
  /// lowering with the hip tapir target.
  std::string hipFeatures;

  /// Absolute paths to the bitcode files that are to be provided to the AMDGPU
  /// backend when the hip tapir target is enabled. The contents of the list are
  /// computed by the driver based on @ref hipArch and any other target-specific
  /// options that were provided.
  llvm::SmallVector<std::string, 4> hipRuntimeBCFiles;

  /// The value of the sramecc feature. The map of values to target features is
  /// as follows:
  ///
  ///     MaybeBool::Off  sramecc-
  ///     MaybeBool::On   sramecc+
  ///     MaybeBool::Any  Value of sramecc is unspecified.
  ///
  MaybeBool hipSRAMECC = defaultHipSRAMECC;

  /// The value of the xnack feature. The map of values to target features is
  /// as follows:
  ///
  ///     MaybeBool::Off  sramecc-
  ///     MaybeBool::On   sramecc+
  ///     MaybeBool::Any  Value of sramecc is unspecified.
  ///
  MaybeBool hipXnack = defaultHipXnack;

  /// Path to the OpenCilk ABI bitcode file. This will only be non-empty if the
  /// OpenCilk tapir target is enabled.
  std::string openCilkRuntimeBCFile;

public:
  KitOptions()
      : kitsuneFrontend(false), kokkos(false), kokkosNoInit(false),
        stripmineLoops(defaultStripmineLoops), gpuPrefetch(defaultGPUPrefetch) {
  }

  /// @{
  /// Setters for options not directly connected to a specific tapir target.
  void setKitsuneFrontend(bool kitsuneFrontend = true) {
    this->kitsuneFrontend = kitsuneFrontend;
  }

  void setKokkos(bool kokkos = true) { this->kokkos = kokkos; }

  void setKokkosNoInit(bool kokkosNoInit = true) {
    this->kokkosNoInit = kokkosNoInit;
  }

  void setTTID(llvm::TTID tt) { this->tt = tt; }

  void setTTPlugin(StringRef path) { this->ttPlugin = path; }

  void setStripmineLoops(bool stripmineLoops = true) {
    this->stripmineLoops = stripmineLoops;
  }
  /// @}

  /// @{
  /// Setters common to the GPU-specific tapir targets.
  void setFixedThreadsPerBlock(unsigned threadsPerBlock) {
    this->fixedThreadsPerBlock = threadsPerBlock;
  }

  void setMaxThreadsPerBlock(unsigned threadsPerBlock) {
    this->maxThreadsPerBlock = threadsPerBlock;
  }

  void setGPUPrefetch(bool prefetch) { this->gpuPrefetch = prefetch; }

  void setLLD(StringRef lld) { this->lld = lld; }
  /// @}

  /// @{
  /// Setters for options related to the cuda tapir target.
  void setCudaArch(StringRef arch) { this->cudaArch = arch; }

  void setCudaVirtArch(StringRef arch) { this->cudaVirtArch = arch; }

  void setCudaFeatures(StringRef features) { this->cudaFeatures = features; }

  void setCudaRuntimeBCFile(StringRef file) { this->cudaRuntimeBCFile = file; }
  /// @}

  /// @{
  /// Setters for options related to the hip tapir target.
  void setHipArch(StringRef arch) { this->hipArch = arch; }

  void setHipFeatures(StringRef features) { this->hipFeatures = features; }

  void setHipSramECC(MaybeBool ecc) { this->hipSRAMECC = ecc; }

  void setHipXnack(MaybeBool xnack) { this->hipXnack = xnack; }

  void addHipRuntimeBCFile(StringRef file) {
    this->hipRuntimeBCFiles.push_back(file.str());
  }
  /// @}

  /// @{
  /// Setters for options related to the opencilk tapir target.
  void setOpenCilkRuntimeBCFile(StringRef path) {
    this->openCilkRuntimeBCFile = path;
  }
  /// @}

  bool isKitsuneFrontend() const { return kitsuneFrontend; }

  bool hasTTID() const { return tt.has_value(); }

  bool hasTTPlugin() const { return ttPlugin.size(); }

  bool getKokkos() const { return kokkos; }

  bool getKokkosNoInit() const { return kokkosNoInit; }

  bool getStripmineLoops() const { return stripmineLoops; }

  std::optional<TTID> getTTID() const { return tt; }

  /// Get the TTID from the options, or a default value.
  /// FIXME: This is *NOT* to be widely used. The presence of this method
  /// implies that there is a "default" tapir target. We need to carefully
  /// consider the implications of this decision.
  TTID getTTIDOr(TTID defawlt) const { return tt.value_or(defawlt); }

  StringRef getTTPlugin() const { return ttPlugin; }

  unsigned getFixedThreadsPerBlock() const { return fixedThreadsPerBlock; }

  unsigned getMaxThreadsPerBlock() const { return maxThreadsPerBlock; }

  bool getGPUPrefetch() const { return gpuPrefetch; }

  StringRef getLLD() const { return lld; }

  StringRef getCudaArch() const { return cudaArch; }

  StringRef getCudaVirtArch() const { return cudaVirtArch; }

  StringRef getCudaFeatures() const { return cudaFeatures; }

  StringRef getCudaRuntimeBCFile() const { return cudaRuntimeBCFile; }

  StringRef getHipArch() const { return hipArch; }

  MaybeBool getHipSRAMECC() const { return hipSRAMECC; }

  MaybeBool getHipXnack() const { return hipXnack; }

  StringRef getHipFeatures() const { return hipFeatures; }

  ArrayRef<std::string> getHipRuntimeBCFiles() const {
    return hipRuntimeBCFiles;
  }

  StringRef getOpenCilkRuntimeBCFile() const { return openCilkRuntimeBCFile; }
};

/// @}

} // namespace driver

} // namespace llvm

#endif // KITSUNE_CORE_KIT_OPTIONS_H
