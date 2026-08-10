//===- runtimes.h - Convenience header for runtime includes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a convenience header that contains a number of declarations, includes
// and other boilerplate for the various runtimes. This is intended to make it
// a bit easier to add a new runtime to Kitsune.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_RUNTIMES_H
#define KITRT_RUNTIMES_H

#include <tuple>

// These are runtimes that are always built, so we unconditionally include the
// headers.
#include "openmp/context.h"
#include "pthreads/context.h"
#include "timer/context.h"

#ifdef KITSUNE_CUDA_ENABLED
#include "cuda/kitcuda.h"
#endif // KITSUNE_CUDA_ENABLED

#ifdef KITSUNE_HIP_ENABLED
#include "hip/kithip.h"
#endif // KITSUNE_HIP_ENABLED

#ifdef KITSUNE_OPENCILK_ENABLED
#include "opencilk/context.h"
#endif // KITSUNE_OPENCILK_ENABLED

#ifdef KITSUNE_PAPI_ENABLED
#include "papi/context.h"
#endif // KITSUNE_PAPI_ENABLED

#ifdef KITSUNE_QTHREADS_ENABLED
#include "qthreads/context.h"
#endif // KITSUNE_QTHREADS_ENABLED

namespace kitrt {

// ---------------------------------- IMPORTANT --------------------------------
//
// When adding a new runtime, a declaration for its singleton context object
// must be added to the forward declarations below
//
// -----------------------------------------------------------------------------
//
// In any given build configuration, one or more of the constituent runtimes may
// not be built. In such cases, we exclude the headers of those runtimes, so a
// declaration for the type will not be available. Failing to provide a
// placeholder in such cases will cause the build to fail.
//
class CudaContext;     // Context for the cuda runtime
class HipContext;      // Context for the hip runtime
class OpenCilkContext; // Context for the opencilk runtime
class OpenMPContext;   // Context for the openmp runtime
class PAPIContext;     // Context for the PAPI support runtime
class PthreadsContext; // Context for the pthreads runtime
class QthreadsContext; // Context for the qthreads runtime
class TimerContext;    // Context for the timer support runtime

// ---------------------------------- IMPORTANT --------------------------------
//
// When a forward declaration is added to the list above, it should also be
// added to the type declaration below.
//
// -----------------------------------------------------------------------------
//
// A tuple of all the known runtimes. This is used in the global singleton
// context that owns all the contexts for the individual runtime as well as any
// shared state. The order in which the objects appear below is not significant.
// They have been sorted alphabetically for convenience.
//
using ContextsTuple =
    std::tuple<CudaContext *,     // Context for the cuda runtime
               HipContext *,      // Context for the hip runtime
               OpenCilkContext *, // Context for the opencilk runtime
               OpenMPContext *,   // Context for the openmp runtime
               PAPIContext *,     // Context for the PAPI support runtime
               PthreadsContext *, // Context for the pthreads runtime
               QthreadsContext *, // Context for the qthreads runtime
               TimerContext *>;   // Context for the timer support runtime

} // namespace kitrt

#endif // KITRT_RUNTIMES_H
