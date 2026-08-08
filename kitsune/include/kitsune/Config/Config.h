//===- Config.h - Query Kitsune build configuration ------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to query Kitsune's build configuration.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CONFIG_CONFIG_H
#define KITSUNE_CONFIG_CONFIG_H

#include "kitsune/Config/config.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

namespace detail {

using TTIDInitList = std::initializer_list<TTID>;
static constexpr TTIDInitList enabledTTs = {KITSUNE_ENABLED_TTIDS};
static constexpr TTIDInitList enabledCPUTTs = {KITSUNE_ENABLED_CPU_TTIDS};
static constexpr TTIDInitList enabledGPUTTs = {KITSUNE_ENABLED_GPU_TTIDS};
static constexpr TTIDInitList enabledEmbBCTTs = {KITSUNE_ENABLED_EMB_BC_TTIDS};
static constexpr TTIDInitList knownTTList = {KITSUNE_KNOWN_TTIDS};
static constexpr TTIDInitList knownCPUTTs = {KITSUNE_KNOWN_CPU_TTIDS};
static constexpr TTIDInitList knownGPUTTs = {KITSUNE_KNOWN_GPU_TTIDS};
static constexpr TTIDInitList knownEmbBCTTs = {KITSUNE_KNOWN_EMB_BC_TTIDS};
static constexpr TTIDInitList universalTTs = {KITSUNE_UNIVERSAL_TTIDS};

// Is the TTID \p tt in TTIDInitList \p list.
static constexpr bool isInTTIDList(TTID tt, TTIDInitList list) {
  for (TTID id : list)
    if (id == tt)
      return true;
  return false;
}

} // namespace detail

/// \addtogroup kitsune
/// @{

/// Get the name of the C frontend driver (unless this has been overridden at
/// configure-time, this should always return kitcc).
constexpr StringRef kitCFrontend() { return KITSUNE_C_FRONTEND; }

/// Get the name of the C++ frontend driver (unless this has been overridden at
/// configure-time, this should always return kit++).
constexpr StringRef kitCXXFrontend() { return KITSUNE_CXX_FRONTEND; }

/// Get the name of the Fortran frontend driver (unless this has been overridden
/// at configure-time, this should always return kitfc).
constexpr StringRef kitFortranFrontend() { return KITSUNE_Fortran_FRONTEND; }

/// A space-separated string of all languages for which Kitsune frontends exist.
constexpr StringRef kitKnownLangs() { return KITSUNE_KNOWN_LANGS; }

/// A space-separated string of all known Tapir targets, including those that
/// have not been enabled.
constexpr StringRef kitKnownTapirTargets() {
  return KITSUNE_KNOWN_TAPIR_TARGETS;
}

/// The list of all known TTID's, including those that have not been enabled.
/// This may, or *may not* contain all values of the TTID enum.
constexpr ArrayRef<TTID> kitKnownTTs() { return detail::knownTTList; }

/// The list of all known TTID's that generate code for GPU's, including those
/// that have not been enabled.
constexpr ArrayRef<TTID> kitKnownCPUTTs() { return detail::knownCPUTTs; }

/// The list of all known TTID's that generate code for GPU's, including those
/// that have not been enabled.
constexpr ArrayRef<TTID> kitKnownGPUTTs() { return detail::knownGPUTTs; }

/// The list of all known TTID's that generate embedded bitcode, including those
/// that have not been enabled.
constexpr ArrayRef<TTID> kitKnownEmbBCTTs() { return detail::knownEmbBCTTs; }

/// The list of universal TTID's - i.e. those that are always enabled.
constexpr ArrayRef<TTID> kitUniversalTTs() { return detail::universalTTs; }

//------------------------------------------------------------------------------

/// The major version of this build.
constexpr unsigned kitMajorVersion() { return KITSUNE_VERSION_MAJOR; }

/// The minor version of this build.
constexpr unsigned kitMinorVersion() { return KITSUNE_VERSION_MINOR; }

/// The patch version of this build.
constexpr unsigned kitPatchVersion() { return KITSUNE_VERSION_PATCH; }

/// The package version of this build.
constexpr StringRef kitPackageVersion() { return KITSUNE_PACKAGE_VERSION; }

/// The default GCC installation directory to use. If one was not explicitly
/// specified at configure-time, this will return an empty string.
constexpr StringRef kitGCCInstallDir() { return KITSUNE_GCC_INSTALL_DIR; }

/// A space-separated string with the Tapir targets that have been enabled in
/// the build.
constexpr StringRef kitEnabledTapirTargets() {
  return KITSUNE_ENABLED_TAPIR_TARGETS;
}

/// The list of TTID's that have been enabled in this build.
constexpr ArrayRef<TTID> kitEnabledTTs() { return detail::enabledTTs; }

/// The list of enabled TTID's that generate CPU code.
constexpr ArrayRef<TTID> kitEnabledCPUTTs() { return detail::enabledCPUTTs; }

/// The list of enabled TTID's that generate GPU code.
constexpr ArrayRef<TTID> kitEnabledGPUTTs() { return detail::enabledGPUTTs; }

/// The the list of enabled TTID's that generate embedded bitcode.
constexpr ArrayRef<TTID> kitEnabledEmbBCTTs() {
  return detail::enabledEmbBCTTs;
}

/// Has the C frontend been built.
constexpr bool kitCEnabled() { return KITSUNE_C_ENABLED; }

/// Has the C++ frontend been built.
constexpr bool kitCXXEnabled() { return KITSUNE_CXX_ENABLED; }

/// Has the Fortran frontend been built.
constexpr bool kitFortranEnabled() { return KITSUNE_Fortran_ENABLED; }

/// A space-separate dstring of all enabled languages.
constexpr StringRef kitEnabledLangs() { return KITSUNE_ENABLED_LANGS; }

//------------------------------------------------------------------------------

/// The base name of Kitsune's runtime library that is distributed as a dynamic
/// shared object. This does not include any file extensions such as .so and
/// .dll, nor any prefix such as "lib".
constexpr StringRef kitRuntimeDSOLibName() { return KITRT_DSO; }

/// The base name of Kitsune's runtime library that is distributed as a static
/// archive. This does not include any file extensions such as .a, nor any
/// prefix such as "lib".
constexpr StringRef kitRuntimeStaticLibName() { return KITRT_STATIC; }

//------------------------------------------------------------------------------

/// Has Kokkos support been enabled.
constexpr bool kitKokkosEnabled() { return KITSUNE_KOKKOS_ENABLED; }

//------------------------------------------------------------------------------

/// Get the install directory of the cuda toolkit used in this build.
constexpr StringRef kitCudaPrefix() { return KITSUNE_CUDA_PREFIX; }

/// The major version of the cuda toolkit used in this build.
constexpr unsigned kitCudaMajorVersion() { return KITSUNE_CUDA_VERSION_MAJOR; }

/// The minor version of the cuda toolkit used in this build.
constexpr unsigned kitCudaMinorVersion() { return KITSUNE_CUDA_VERSION_MINOR; }

/// The patch version of the cuda toolkit used in this build.
constexpr unsigned kitCudaPatchVersion() { return KITSUNE_CUDA_VERSION_PATCH; }

/// The package version of the cuda toolkit used in this build.
constexpr StringRef kitCudaPackageVersion() {
  return KITSUNE_CUDA_VERSION_STRING;
}

/// Path to cuda ptxas executable used in this build.
constexpr StringRef kitCudaPtxas() { return KITSUNE_CUDA_PTXAS; }

/// Path to cuda's fatbinary executable used in this build.
constexpr StringRef kitCudaFatbinary() { return KITSUNE_CUDA_FATBINARY; }

/// The default NVIDIA GPU architecture for which to generate code.
constexpr StringRef kitCudaDefaultArch() { return KITSUNE_CUDA_ARCH_DEFAULT; }

/// The name of the symbol that will eventually contain the NVIDIA GPU device
/// code.
constexpr StringRef kitCudaFatbinName() { return KITSUNE_CUDA_FATBIN_NAME; }

//------------------------------------------------------------------------------

/// Get the install directory of AMD's ROCm used in this build.
constexpr StringRef kitHipPrefix() { return KITSUNE_HIP_PREFIX; }

/// The major version of AMD's ROCm used in this build.
constexpr unsigned kitHipMajorVersion() { return KITSUNE_HIP_VERSION_MAJOR; }

/// The minor version of AMD's ROCm used in this build.
constexpr unsigned kitHipMinorVersion() { return KITSUNE_HIP_VERSION_MINOR; }

/// The patch version of AMD's ROCm used in this build.
constexpr unsigned kitHipPatchVersion() { return KITSUNE_HIP_VERSION_PATCH; }

/// The package version of AMD's ROCm used in this build.
constexpr StringRef kitHipPackageVersion() {
  return KITSUNE_HIP_VERSION_STRING;
}

/// The path to the directory containing AMDGPU device bitcode files used in
/// this build.
constexpr StringRef kitHipDeviceBitcodeDir() { return KITSUNE_HIP_BITCODE_DIR; }

/// The default AMD GPU architecture for which to generate code.
constexpr StringRef kitHipDefaultArch() { return KITSUNE_HIP_ARCH_DEFAULT; }

/// The name of the symbol that will eventually contain AMD GPU device code.
constexpr StringRef kitHipFatbinName() { return KITSUNE_HIP_FATBIN_NAME; }

//------------------------------------------------------------------------------

/// @}

} // namespace llvm

#endif // KITSUNE_CONFIG_CONFIG_H
