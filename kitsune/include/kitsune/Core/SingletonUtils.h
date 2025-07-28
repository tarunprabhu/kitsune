//===- SingletonUtils.h - Utilities for Kitsune's singletons ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kitsune uses some singletons to interface correctly with its own runtime, and
// the external runtimes used by some tapir targets (such as libcuda and
// libamdhip64). This is a central location where the names of these singletons
// are defined. Several convenience functions are also provided.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_SINGLETON_UTILS_H
#define KITSUNE_CORE_SINGLETON_UTILS_H

#include "kitsune/Config/config.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class GlobalVariable;
class Module;

// Utilities for singletons used by the cuda tapir target.
// @{

// The name of the global variable that will contain the linked NVIDIA fat
// binary.
constexpr StringLiteral cudaFatbinName = KITSUNE_CUDA_FB_NAME;

// The section containing the cuda fat binary.
constexpr StringLiteral cudaFatbinSection = KITSUNE_CUDA_FB_SECTION;

// The section containing the cuda fat binary "bundle". The bundle consists of
// a magic number, version number and the fat binary. NVIDIA's tools look for
// the fat binary in this section.
constexpr StringLiteral cudaBundleSection = KITSUNE_CUDA_BUNDLE_SECTION;

// The name of the section containing relocatable cuda code. This is where
// kit-fblink looks for embedded device code object files.
constexpr StringLiteral cudaDeviceCodeSection = KITSUNE_CUDA_EMB_CODE_SECTION;

// @}

// Utilities for singletons used by the hip tapir target.
// @{

// The name of the global variable that will contain the linked AMDGPU fat
// binary.
constexpr StringLiteral hipFatbinName = KITSUNE_HIP_FB_NAME;

// The section containing the cuda fat binary.
constexpr StringLiteral hipFatbinSection = KITSUNE_HIP_FB_SECTION;

// The section containing the hip fat binary "bundle". The bundle consists of
// a magic number, version number and the fat binary. AMD's tools look for the
// fat binary in this section.
constexpr StringLiteral hipBundleSection = KITSUNE_HIP_BUNDLE_SECTION;

// The name of the section containing relocatable hip code. This is where
// kit-fblink looks for embedded device code object files.
constexpr StringLiteral hipDeviceCodeSection = KITSUNE_HIP_EMB_CODE_SECTION;

// @}

/// Get the name of the singleton global variable that will contain the fat
/// binary for the given tapir target.
StringLiteral getFatbinName(TTID tt);

/// Get the name of the section containing the singleton fat binary global
/// varible.
StringLiteral getFatbinSection(TTID tt);

/// Get the global variable created by a previous call to @ref
/// createSingletonFBGlobal with the given tapir target if one exists.
GlobalVariable *getSingletonFBGlobal(TTID tt, Module &m);

/// Create a global variable which will contains the fully linked fat binary.
/// This will have external linkage and no initializer since it will only become
/// available at link time.
///
/// @param tt The ID of the tapir target that is creating this embedded bitcode
/// @param m The host module into which the global variable will be created
/// @returns The newly created global variable
GlobalVariable *createSingletonFBGlobal(TTID tt, Module &m);

} // namespace llvm

#endif // KITSUNE_CORE_SINGLETON_UTILS_H
