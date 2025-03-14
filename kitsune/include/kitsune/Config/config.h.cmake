/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */

/* This generated file is for internal use. Do not include it from headers. */

#ifndef KITSUNE_CONFIG_H
#define KITSUNE_CONFIG_H

// General configuration
#define KITSUNE_LIBNAME "${KITSUNE_LIBNAME}"
#define KITSUNE_LIBNAME_STATIC "${KITSUNE_LIBNAME_STATIC}"

// The address space for Kitsune's mobile pointers. This is a lousy workaround
// because there is no way to have attributed types in LLVM, but we want to be
// able to identify pointers to data that may be moved between host and device
// memory when GPU tapir targets are enabled. LLVM has a maximum of 24-bits for
// the address space, clang has 22 (23 usually, but we stole a bit in Kitsune -
// ironically, for the mobile attribute). Just in case, don't use more than
// 16 bits for this.
constexpr unsigned KITSUNE_ADDRSPACE = 67;
static_assert(KITSUNE_ADDRSPACE <= 0xFFFF &&
              "Kitsune's address space must occupy no more than 16 bits");

// When using a fixed number of threads per block, the maximum value that can be
// specified.
constexpr unsigned KITSUNE_MAX_FIXED_THREADS_PER_BLOCK = 1024;

// Kitsune language support that has been enabled
#cmakedefine01 KITSUNE_C_ENABLED
#cmakedefine01 KITSUNE_CXX_ENABLED
#cmakedefine01 KITSUNE_Fortran_ENABLED

// The names of the Kitsune frontends
#define KITSUNE_C_FRONTEND "${KITSUNE_C_FRONTEND}"
#define KITSUNE_CXX_FRONTEND "${KITSUNE_CXX_FRONTEND}"
#define KITSUNE_Fortran_FRONTEND "${KITSUNE_Fortran_FRONTEND}"
#define KITSUNE_ALL_FRONTENDS                                                  \
  KITSUNE_C_FRONTEND " " KITSUNE_CXX_FRONTEND " " KITSUNE_Fortran_FRONTEND

// Kokkos configuration
#cmakedefine01 KITSUNE_KOKKOS_ENABLED

#define KITSUNE_KOKKOS_EXTRA_PREPROCESSOR_FLAGS                                \
  "${KITSUNE_KOKKOS_EXTRA_PREPROCESSOR_FLAGS}"
#define KITSUNE_KOKKOS_EXTRA_COMPILER_FLAGS                                    \
  "${KITSUNE_KOKKOS_EXTRA_COMPILER_FLAGS}"
#define KITSUNE_KOKKOS_EXTRA_LINKER_FLAGS "${KITSUNE_KOKKOS_EXTRA_LINKER_FLAGS}"

// Cuda configuration
#cmakedefine01 KITSUNE_CUDA_ENABLED

#define KITSUNE_CUDA_EXTRA_PREPROCESSOR_FLAGS                                  \
  "${KITSUNE_CUDA_EXTRA_PREPROCESSOR_FLAGS}"
#define KITSUNE_CUDA_EXTRA_COMPILER_FLAGS "${KITSUNE_CUDA_EXTRA_COMPILER_FLAGS}"
#define KITSUNE_CUDA_EXTRA_LINKER_FLAGS "${KITSUNE_CUDA_EXTRA_LINKER_FLAGS}"

#define KITSUNE_CUDA_PREFIX "${KITSUNE_CUDA_PREFIX}"
#define KITSUNE_CUDA_VERSION_MAJOR ${KITSUNE_CUDA_VERSION_MAJOR}
#define KITSUNE_CUDA_VERSION_MINOR ${KITSUNE_CUDA_VERSION_MINOR}
#define KITSUNE_CUDA_VERSION_PATCH ${KITSUNE_CUDA_VERSION_PATCH}
#define KITSUNE_CUDA_VERSION_STRING "${KITSUNE_CUDA_VERSION_STRING}"
#define KITSUNE_CUDA_BINARY_DIR "${KITSUNE_CUDA_BINARY_DIR}"
#define KITSUNE_CUDA_LIBRARY_DIR "${KITSUNE_CUDA_LIBRARY_DIR}"
#define KITSUNE_CUDA_LIBCUDA_DIR "${KITSUNE_CUDA_LIBCUDA_DIR}"
#define KITSUNE_CUDA_LIBDEVICE_DIR "${KITSUNE_CUDA_LIBDEVICE_DIR}"
#define KITSUNE_CUDA_LIBDEVICE_BC "${KITSUNE_CUDA_LIBDEVICE_BC}"
#define KITSUNE_CUDA_PTXAS "${KITSUNE_CUDA_PTXAS}"
#define KITSUNE_CUDA_FATBINARY "${KITSUNE_CUDA_FATBINARY}"
#define KITSUNE_CUDA_ARCH_DEFAULT "${KITSUNE_CUDA_ARCH_DEFAULT}"

#define KITSUNE_CUDA_LIBNAME_CUDA "${KITSUNE_CUDA_LIBNAME_CUDA}"
#define KITSUNE_CUDA_LIBNAME_CUDART "${KITSUNE_CUDA_LIBNAME_CUDART}"
#define KITSUNE_CUDA_LIBNAME_CUDART_STATIC "${KITSUNE_CUDA_LIBNAME_CUDART_STATIC}"

// The name of the symbol containing the cuda device code (the fat binary)
#define KITSUNE_CUDA_FATBIN_NAME "__kitsune_fatbin_cuda"

// The name of the symbol containing the cuda device bitcode. This is what is
// compiled to form the fat binary.
#define KITSUNE_CUDA_FATBC_NAME "__kitsune_fatbc_cuda"

// Hip configuration
#cmakedefine01 KITSUNE_HIP_ENABLED

#define KITSUNE_HIP_EXTRA_PREPROCESSOR_FLAGS                                   \
  "${KITSUNE_HIP_EXTRA_PREPROCESSOR_FLAGS}"
#define KITSUNE_HIP_EXTRA_COMPILER_FLAGS "${KITSUNE_HIP_EXTRA_COMPILER_FLAGS}"
#define KITSUNE_HIP_EXTRA_LINKER_FLAGS "${KITSUNE_HIP_EXTRA_LINKER_FLAGS}"

#define KITSUNE_HIP_PREFIX "${KITSUNE_HIP_PREFIX}"
#define KITSUNE_HIP_VERSION_MAJOR ${KITSUNE_HIP_VERSION_MAJOR}
#define KITSUNE_HIP_VERSION_MINOR ${KITSUNE_HIP_VERSION_MINOR}
#define KITSUNE_HIP_VERSION_PATCH ${KITSUNE_HIP_VERSION_PATCH}
#define KITSUNE_HIP_VERSION_STRING "${KITSUNE_HIP_VERSION_STRING}"
#define KITSUNE_HIP_LIBRARY_DIR "${KITSUNE_HIP_LIBRARY_DIR}"
#define KITSUNE_HIP_BITCODE_DIR "${KITSUNE_HIP_BITCODE_DIR}"
#define KITSUNE_HIP_ARCH_DEFAULT "${KITSUNE_HIP_ARCH_DEFAULT}"

#define KITSUNE_HIP_LIBNAME_AMDHIP "${KITSUNE_HIP_LIBNAME_AMDHIP}"

// The name of the symbol containing the hip device code (the fat binary)
#define KITSUNE_HIP_FATBIN_NAME "__kitsune_fatbin_hip"

// The name of the symbol containing the hip device bitcode. This is what is
// compiled to form the fat binary.
#define KITSUNE_HIP_FATBC_NAME  "__kitsune_fatbc_hip"

// Lambda configuration
#cmakedefine01 KITSUNE_LAMBDA_ENABLED

#define KITSUNE_LAMBDA_EXTRA_PREPROCESSOR_FLAGS                                \
  "${KITSUNE_LAMBDA_EXTRA_PREPROCESSOR_FLAGS}"
#define KITSUNE_LAMBDA_EXTRA_COMPILER_FLAGS                                    \
  "${KITSUNE_LAMBDA_EXTRA_COMPILER_FLAGS}"
#define KITSUNE_LAMBDA_EXTRA_LINKER_FLAGS "${KITSUNE_LAMBDA_EXTRA_LINKER_FLAGS}"

// OMPTask configuration
#cmakedefine01 KITSUNE_OMPTASK_ENABLED

#define KITSUNE_OMPTASK_EXTRA_PREPROCESSOR_FLAGS                               \
  "${KITSUNE_OMPTASK_EXTRA_PREPROCESSOR_FLAGS}"
#define KITSUNE_OMPTASK_EXTRA_COMPILER_FLAGS                                   \
  "${KITSUNE_OMPTASK_EXTRA_COMPILER_FLAGS}"
#define KITSUNE_OMPTASK_EXTRA_LINKER_FLAGS                                     \
  "${KITSUNE_OMPTASK_EXTRA_LINKER_FLAGS}"

// OpenCilk configuration
#cmakedefine01 KITSUNE_OPENCILK_ENABLED

#define KITSUNE_OPENCILK_EXTRA_PREPROCESSOR_FLAGS                              \
  "${KITSUNE_OPENCILK_EXTRA_PREPROCESSOR_FLAGS}"
#define KITSUNE_OPENCILK_EXTRA_COMPILER_FLAGS                                  \
  "${KITSUNE_OPENCILK_EXTRA_COMPILER_FLAGS}"
#define KITSUNE_OPENCILK_EXTRA_LINKER_FLAGS                                    \
  "${KITSUNE_OPENCILK_EXTRA_LINKER_FLAGS}"

// OpenMP configuration
#cmakedefine01 KITSUNE_OPENMP_ENABLED

#define KITSUNE_OPENMP_EXTRA_PREPROCESSOR_FLAGS                                \
  "${KITSUNE_OPENMP_EXTRA_PREPROCESSOR_FLAGS}"
#define KITSUNE_OPENMP_EXTRA_COMPILER_FLAGS                                    \
  "${KITSUNE_OPENMP_EXTRA_COMPILER_FLAGS}"
#define KITSUNE_OPENMP_EXTRA_LINKER_FLAGS "${KITSUNE_OPENMP_EXTRA_LINKER_FLAGS}"

// Qthreads configuration
#cmakedefine01 KITSUNE_QTHREADS_ENABLED

#define KITSUNE_QTHREADS_EXTRA_PREPROCESSOR_FLAGS                              \
  "${KITSUNE_QTHREADS_EXTRA_PREPROCESSOR_FLAGS}"
#define KITSUNE_QTHREADS_EXTRA_COMPILER_FLAGS                                  \
  "${KITSUNE_QTHREADS_EXTRA_COMPILER_FLAGS}"
#define KITSUNE_QTHREADS_EXTRA_LINKER_FLAGS                                    \
  "${KITSUNE_QTHREADS_EXTRA_LINKER_FLAGS}"

// Realm configuration
#cmakedefine01 KITSUNE_REALM_ENABLED

#define KITSUNE_REALM_EXTRA_PREPROCESSOR_FLAGS                                 \
  "${KITSUNE_REALM_EXTRA_PREPROCESSOR_FLAGS}"
#define KITSUNE_REALM_EXTRA_COMPILER_FLAGS                                     \
  "${KITSUNE_REALM_EXTRA_COMPILER_FLAGS}"
#define KITSUNE_REALM_EXTRA_LINKER_FLAGS "${KITSUNE_REALM_EXTRA_LINKER_FLAGS}"

// All known Tapir targets, including those that have not been enabled.
#define KITSUNE_ALL_TAPIR_TARGETS "${KITSUNE_ALL_TAPIR_TARGETS_STR}"

// The Tapir targets that have been enabled in this build
#define KITSUNE_ENABLED_TAPIR_TARGETS "${KITSUNE_ENABLED_TAPIR_TARGETS_STR}"

#endif // KITSUNE_CONFIG_H
