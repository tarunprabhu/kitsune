//===- config.h.cmake - Configuration variables from cmake  ----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Configuration variables initialized by cmake.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CONFIG_H
#define KITSUNE_CONFIG_H

#define KITSUNE_VERSION_MAJOR @KITSUNE_VERSION_MAJOR@
#define KITSUNE_VERSION_MINOR @KITSUNE_VERSION_MINOR@
#define KITSUNE_VERSION_PATCH @KITSUNE_VERSION_PATCH@
#define KITSUNE_PACKAGE_VERSION "@KITSUNE_PACKAGE_VERSION@"

//------------------------------------------------------------------------------

// The default GCC installation directory to use. If this is non-null, Kitsune
// will not scan for a GCC installation. This value can be overridden by passing
// --gcc-install-dir on the command line.
#define KITSUNE_GCC_INSTALL_DIR "@KITSUNE_GCC_INSTALL_DIR@"

//------------------------------------------------------------------------------

// The names of the Kitsune frontends.
#define KITSUNE_C_FRONTEND "@KITSUNE_C_FRONTEND@"
#define KITSUNE_CXX_FRONTEND "@KITSUNE_CXX_FRONTEND@"
#define KITSUNE_Fortran_FRONTEND "@KITSUNE_Fortran_FRONTEND@"

// A space-separated string of all languages for which Kitsune frontends exist.
#define KITSUNE_KNOWN_LANGS "@KITSUNE_KNOWN_LANGS_STR@"

// Kitsune language support that has been enabled.
#cmakedefine01 KITSUNE_C_ENABLED
#cmakedefine01 KITSUNE_CXX_ENABLED
#cmakedefine01 KITSUNE_Fortran_ENABLED

// A space-separate dstring of all enabled languages.
#define KITSUNE_ENABLED_LANGS "@KITSUNE_ENABLED_LANGS_STR@"

//------------------------------------------------------------------------------

// A space-separated string of all known tapir targets, including those that
// have not been enabled.
#define KITSUNE_KNOWN_TAPIR_TARGETS "@KITSUNE_KNOWN_TAPIR_TARGETS_STR@"

// A space-separated string with the tapir targets that have been enabled in
// this build.
#define KITSUNE_ENABLED_TAPIR_TARGETS "@KITSUNE_ENABLED_TAPIR_TARGETS_STR@"

// A comma-separated list of all known TTID's. This may not contain all values
// of the TTID enum. Rather, it only contains those TTID's that are known to
// the build system. For instance, TTID::Nolo will never be on this list since
// it is not a "true" tapir target. TTID::Lambda will not appear in this list
// either because, at the time of writing this comment, Kitsune does not support
// it. This will include TTID's of targets that are not enabled in this build.
// This is *NOT* a string. It can only be used within an initializer list.
#define KITSUNE_KNOWN_TTIDS @KITSUNE_KNOWN_TTIDS@

// A comma-separated list of all TTID's that generate GPU code. This will
// include TTID's of targets that are not enabled in this build.
// This is *NOT* a string. It can only be used within an initializer list.
#define KITSUNE_KNOWN_GPU_TTIDS @KITSUNE_KNOWN_GPU_TTIDS@

// A comma-separated list of all TTID's that generate embedded bitcode. This
// will include TTID's of targets that are not enabled in this build.
// This is *NOT* a string. It can only be used within an initializer list.
#define KITSUNE_KNOWN_EMB_BC_TTIDS @KITSUNE_KNOWN_EMB_BC_TTIDS@

// A comma-separated list of tapir targets that have been enabled in this build.
// This is *NOT* a string. It can only be used within an initializer list.
#define KITSUNE_ENABLED_TTIDS @KITSUNE_ENABLED_TTIDS@

// A comma-separated list of tapir targets that generate GPU code and have been
// enabled in this build.
// This is *NOT* a string. It can only be used within an initializer list.
#define KITSUNE_ENABLED_GPU_TTIDS @KITSUNE_ENABLED_GPU_TTIDS@

// A comma-separated list of tapir targets that generated embedded bitcode and
// have been enabled in this build.
// This is *NOT* a string. It can only be used within an initializer list.
#define KITSUNE_ENABLED_EMB_BC_TTIDS @KITSUNE_ENABLED_EMB_BC_TTIDS@

// A comma-separated list of tapir targets that are always enabled.
// This is *NOT* a string. It can only be used within an initializer list.
#define KITSUNE_UNIVERSAL_TTIDS @KITSUNE_UNIVERSAL_TTIDS@

//------------------------------------------------------------------------------

// The basename of the dynamic library containing Kitsune's runtime. This does
// not contain any platform-specific prefixes or extensions.
#define KITRT_DSO "@KITRT_DSO@"

// The basename of the static archive containing Kitsune's runtime. This does
// not contain any platform-specific prefixes or extensions.
#define KITRT_STATIC "@KITRT_STATIC@"

//------------------------------------------------------------------------------

// Has kokkos support been enabled.
#cmakedefine01 KITSUNE_KOKKOS_ENABLED

//------------------------------------------------------------------------------

// Has the cuda tapir target been enabled.
#cmakedefine01 KITSUNE_CUDA_ENABLED

#define KITSUNE_CUDA_PREFIX "@KITSUNE_CUDA_PREFIX@"
#define KITSUNE_CUDA_VERSION_MAJOR @KITSUNE_CUDA_VERSION_MAJOR@
#define KITSUNE_CUDA_VERSION_MINOR @KITSUNE_CUDA_VERSION_MINOR@
#define KITSUNE_CUDA_VERSION_PATCH @KITSUNE_CUDA_VERSION_PATCH@
#define KITSUNE_CUDA_VERSION_STRING "@KITSUNE_CUDA_VERSION_STRING@"
#define KITSUNE_CUDA_PTXAS "@KITSUNE_CUDA_PTXAS@"
#define KITSUNE_CUDA_FATBINARY "@KITSUNE_CUDA_FATBINARY@"
#define KITSUNE_CUDA_ARCH_DEFAULT "@KITSUNE_CUDA_ARCH_DEFAULT@"

// The name of the symbol containing the cuda device code (the fat binary).
#define KITSUNE_CUDA_FATBIN_NAME "__kitsune_fatbin_cuda"

//------------------------------------------------------------------------------

// Has the custom tapir target been enabled.
#cmakedefine01 KITSUNE_CUSTOM_ENABLED

//------------------------------------------------------------------------------

// Has the hip tapir target been enabled.
#cmakedefine01 KITSUNE_HIP_ENABLED

#define KITSUNE_HIP_PREFIX "@KITSUNE_HIP_PREFIX@"
#define KITSUNE_HIP_VERSION_MAJOR @KITSUNE_HIP_VERSION_MAJOR@
#define KITSUNE_HIP_VERSION_MINOR @KITSUNE_HIP_VERSION_MINOR@
#define KITSUNE_HIP_VERSION_PATCH @KITSUNE_HIP_VERSION_PATCH@
#define KITSUNE_HIP_VERSION_STRING "@KITSUNE_HIP_VERSION_STRING@"
#define KITSUNE_HIP_BITCODE_DIR "@KITSUNE_HIP_BITCODE_DIR@"
#define KITSUNE_HIP_ARCH_DEFAULT "@KITSUNE_HIP_ARCH_DEFAULT@"

// The name of the symbol containing the hip device code (the fat binary).
#define KITSUNE_HIP_FATBIN_NAME "__kitsune_fatbin_hip"

//------------------------------------------------------------------------------

// Has the lambda tapir target been enabled.
#cmakedefine01 KITSUNE_LAMBDA_ENABLED

//------------------------------------------------------------------------------

// Has the omptask tapir target been enabled.
#cmakedefine01 KITSUNE_OMPTASK_ENABLED

//------------------------------------------------------------------------------

// Has the opencilk tapir target been enabled.
#cmakedefine01 KITSUNE_OPENCILK_ENABLED

//------------------------------------------------------------------------------

// Has the openmp tapir target been enabled.
#cmakedefine01 KITSUNE_OPENMP_ENABLED

//------------------------------------------------------------------------------

// Has the pthreads tapir target been enabled.
#cmakedefine01 KITSUNE_PTHREADS_ENABLED

//------------------------------------------------------------------------------

// Has the qthreads tapir target been enabled.
#cmakedefine01 KITSUNE_QTHREADS_ENABLED

//------------------------------------------------------------------------------

// Has the realm tapir target been enabled.
#cmakedefine01 KITSUNE_REALM_ENABLED

//------------------------------------------------------------------------------

// Has the serial tapir target been enabled.
#cmakedefine01 KITSUNE_SERIAL_ENABLED

//------------------------------------------------------------------------------

#endif // KITSUNE_CONFIG_H
