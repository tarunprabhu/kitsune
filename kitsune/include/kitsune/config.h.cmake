/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */
#ifdef KITSUNE_CONFIG_H
#error kitsune config.h should only be included once.
#else
#define KITSUNE_CONFIG_H

/* Is Kokkos support enabled? */
#cmakedefine01 KITSUNE_ENABLE_KOKKOS

/* Is the OpenCilk runtime target enabled? */
#cmakedefine01 TAPIR_ENABLE_OPENCILKRTS_TARGET

/* Is the Realm runtime target enabled? */
#cmakedefine01 TAPIR_ENABLE_REALM_TARGET

/* Is the Qthreads runtime target enabled? */.
#cmakedefine01 TAPIR_ENABLE_QTHREADS_TARGET

/* Is the OMP runtime target enabled? */
#cmakedefine01 TAPIR_ENABLE_OMP_TARGET

/* Is the CUDA runtime target enabled? */
 #cmakedefine01 TAPIR_ENABLE_CUDA_TARGET

/* Is the OpenCL runtime target enabled? */
#cmakedefine01 KITSUNE_ENABLE_OPENCL_TARGET

#endif
