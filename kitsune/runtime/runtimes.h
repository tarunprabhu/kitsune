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

#include "kitsune/Shared/RTID.h"
#include "kitsune/Shared/TypeTraits.h"

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

namespace detail {

// Trait to check if a function returns void. This will work for functions that
// take any number of arguments, including zero.
template <typename T> struct is_return_void : std::false_type {};
template <typename R, typename... Args>
struct is_return_void<R (*)(Args...)>
    : std::bool_constant<std::is_same_v<R, void>> {};
template <typename R, typename C, typename... Args>
struct is_return_void<R (C::*)(Args...)>
    : std::bool_constant<std::is_same_v<R, void>> {};

// Trait to check if a class has a member named `initialize` that takes any
// number of arguments and returns void.
template <typename T, typename = void>
struct has_initialize : std::false_type {};
template <typename T>
struct has_initialize<T, std::void_t<decltype(&T::initialize)>>
    : std::bool_constant<
          std::is_member_function_pointer_v<decltype(&T::initialize)> &&
          is_return_void<decltype(&T::initialize)>::value> {};

// Trait to check if a class has a member named `finalize` that does not take
// any arguments and returns void.
template <typename T, typename = void> struct has_finalize : std::false_type {};
template <typename T>
struct has_finalize<
    T, std::void_t<decltype(static_cast<void (T::*)()>(&T::finalize), void())>>
    : std::true_type {};

// Trait to check if a type is a runtime context. This is is assumed to be true
// if the type has members named initialize and finalize with the following
// signatures.
//
//     void initialize(...)
//     void finalize()
//
// Here `...` indicates that the initialize member may take any number of
// arguments, including zero. These members are expected by the core runtime
// since it must initialize any requested runtimes.
//
// We decided against using a base class mixin since it would have been empty
// anyway. Virtual functions were never an option since the initialize() member
// could have different signatures in the various derived classes.
template <typename T>
static constexpr bool is_context_v =
    has_initialize<T>::value && has_finalize<T>::value;

// Check if a type is a pointer to a context. In any given build, some runtimes
// may be disabled. In such cases, we will still have a pointer, but to an
// incomplete type. Since we know that the rest of the runtime will never try to
// use that type for anything, we assume that it is a valid context pointer.
// Obviously, if the type were to be opaque for some other reason, it is an
// error, but there is no way for us to reasonably diagnose this. If this does
// not raise a compilation error elsewhere in the code, we can expect a very
// unpleasant runtime error.
template <typename T> struct is_context_ptr {
  static constexpr bool value =
      std::is_pointer_v<T> && (!std::is_complete_v<std::remove_pointer_t<T>> ||
                               is_context_v<std::remove_pointer_t<T>>);
};

// Check the contexts tuple. Every type in the tuple must be a pointer to a
// context object. \pref is_context_ptr is used to check the actual type. This
// trait simply uses that trait on every type of the tuple.
template <typename T> struct check_contexts;
template <typename... Types>
struct check_contexts<std::tuple<Types...>>
    : std::conjunction<is_context_ptr<Types>...> {};

// Compile-time helper trait to represent an RTID.
template <RTID RT> using rtid_v = std::integral_constant<RTID, RT>;

} // namespace detail

// ---------------------------------- IMPORTANT --------------------------------
//
// Read the documentation in this section when adding a new
// tapir-target-specific, or support, runtime.

// In any given build configuration, one or more of the constituent runtimes may
// not be built. In such cases, we exclude the headers of those runtimes, so a
// declaration for the type will not be available. Failing to provide a
// placeholder in such cases will cause the build to fail.
class CudaContext;     // Context for the cuda runtime
class HipContext;      // Context for the hip runtime
class OpenCilkContext; // Context for the opencilk runtime
class OpenMPContext;   // Context for the openmp runtime
class PAPIContext;     // Context for the PAPI support runtime
class PthreadsContext; // Context for the pthreads runtime
class QthreadsContext; // Context for the qthreads runtime
class TimerContext;    // Context for the timer support runtime

// A tuple of all the known runtimes. This is used in the global singleton
// context that owns all the contexts for the individual runtime as well as any
// shared state. The order in which the objects appear below is not significant.
// They have been sorted alphabetically for convenience.
using ContextsTuple =
    std::tuple<CudaContext *,     // Context for the cuda runtime
               HipContext *,      // Context for the hip runtime
               OpenCilkContext *, // Context for the opencilk runtime
               OpenMPContext *,   // Context for the openmp runtime
               PAPIContext *,     // Context for the PAPI support runtime
               PthreadsContext *, // Context for the pthreads runtime
               QthreadsContext *, // Context for the qthreads runtime
               TimerContext *>;   // Context for the timer support runtime

// Get the name of the runtime. This is mainly used for logging and error
// messages. We have to resort to a free function and explicit specialization
// because we need this to be available even when a runtime has not been
// enabled. This is the only method that is guaranteed to work even if we only
// have forward declared types.
template <RTID RT> static inline constexpr const char *rtname_v = nullptr;
template <> inline constexpr const char *rtname_v<RT_CUDA> = "cuda";
template <> inline constexpr const char *rtname_v<RT_HIP> = "hip";
template <> inline constexpr const char *rtname_v<RT_OPENCILK> = "opencilk";
template <> inline constexpr const char *rtname_v<RT_OPENMP> = "openmp";
template <> inline constexpr const char *rtname_v<RT_PAPI> = "papi";
template <> inline constexpr const char *rtname_v<RT_PTHREADS> = "pthreads";
template <> inline constexpr const char *rtname_v<RT_QTHREADS> = "qthreads";
template <> inline constexpr const char *rtname_v<RT_TIMER> = "timer";

// If the runtime is CPU-centric that launches parallel threads of execution,
// and provides a function to query the ID of a thread, a specialization for
// this trait should be provided.
template <RTID RT> struct is_cpu_threaded_t : std::false_type {};
template <> struct is_cpu_threaded_t<RT_OPENCILK> : std::true_type {};
template <> struct is_cpu_threaded_t<RT_OPENMP> : std::true_type {};
template <> struct is_cpu_threaded_t<RT_PTHREADS> : std::true_type {};
template <> struct is_cpu_threaded_t<RT_QTHREADS> : std::true_type {};

// clang-format off
// A trait to obtain the type of a context object given the RTID.
template <RTID RT> struct context_t;
template <> struct context_t<RT_CUDA> { using type = CudaContext; };
template <> struct context_t<RT_HIP> { using type = HipContext; };
template <> struct context_t<RT_OPENCILK> { using type = OpenCilkContext; };
template <> struct context_t<RT_OPENMP> { using type = OpenMPContext; };
template <> struct context_t<RT_PAPI> { using type = PAPIContext; };
template <> struct context_t<RT_PTHREADS> { using type = PthreadsContext; };
template <> struct context_t<RT_QTHREADS> { using type = QthreadsContext; };
template <> struct context_t<RT_TIMER> { using type = TimerContext; };
// clang-format on

// A trait to obtain the RTID corresponding to the type of a context object.
template <typename T> struct rtid_v;
template <> struct rtid_v<CudaContext> : detail::rtid_v<RT_CUDA> {};
template <> struct rtid_v<HipContext> : detail::rtid_v<RT_HIP> {};
template <> struct rtid_v<OpenCilkContext> : detail::rtid_v<RT_OPENCILK> {};
template <> struct rtid_v<OpenMPContext> : detail::rtid_v<RT_OPENMP> {};
template <> struct rtid_v<PAPIContext> : detail::rtid_v<RT_PAPI> {};
template <> struct rtid_v<PthreadsContext> : detail::rtid_v<RT_PTHREADS> {};
template <> struct rtid_v<QthreadsContext> : detail::rtid_v<RT_QTHREADS> {};
template <> struct rtid_v<TimerContext> : detail::rtid_v<RT_TIMER> {};

// -----------------------------------------------------------------------------

static_assert(detail::check_contexts<ContextsTuple>::value,
              "All registered contexts must be valid context pointers");

} // namespace kitrt

#endif // KITRT_RUNTIMES_H
