//===- kitrt.cpp - The main interface for Kitsune's runtime ---------------===//
//
// Copyright (c) 2021, Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021. Los Alamos National Security, LLC. This software was
//  produced under U.S. Government contract DE-AC52-06NA25396 for Los
//  Alamos National Laboratory (LANL), which is operated by Los Alamos
//  National Security, LLC for the U.S. Department of Energy. The
//  U.S. Government has rights to use, reproduce, and distribute this
//  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
//  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
//  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
//  derivative works, such modified software should be clearly marked,
//  so as not to confuse it with the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
//  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#include "kitrt.h"
#include "common/env.h"
#include "common/logging.h"
#include "common/unreachable.h"
#include "context.h"
#include "kitsune/Shared/RTInitOptions.h"
#include "runtimes.h"

#ifdef KITSUNE_COLORS_ENABLED
#include <curses.h>
#include <unistd.h>
#endif // KITSUNE_COLORS_ENABLED

#include <algorithm>
#include <vector>

using namespace kitrt;

static bool terminalHasColors() {
  bool colors = false;

#ifdef KITSUNE_COLORS_ENABLED
  // Respect the NO_COLOR environment variable. If it is present, don't use
  // colors. Conversely, if FORCE_COLOR is present, always use colors, even
  // when not writing to a tty. If both are present, NO_COLOR takes precedence,
  // only because NO_COLOR is supported in clang, while FORCE_COLOR isn't. That
  // is a pretty useless reason, but there doesn't seem to be an established
  // convention for this case. For instance, Python gives NO_COLOR precedence,
  // whereas NVIDIA Legate gives preference to FORCE_COLOR.
  if (envContains("NO_COLOR"))
    return false;

  if (envContains("FORCE_COLOR"))
    return true;

  // Don't use colors if stderr is not connected to a terminal
  if (!isatty(STDERR_FILENO))
    return false;

  // Otherwise, do the "sane" thing and use colors only if the terminal supports
  // it. This creates a terminal curses terminal for the sole purpose of
  // querying it for color support.
  SCREEN *scr = newterm(NULL, stderr, stdin);
  colors = has_colors();
  endwin();
  delscreen(scr);
#endif // !KITSUNE_COLORS_ENABLED

  return colors;
}

// Sort the requested runtimes in "dependence order". There are only a few
// dependencies between the runtimes.
//
//   - All runtimes depend on the common runtime, RT_COMMON.
//
//   - If PAPI has been enabled, and per-thread counters are to be measured, the
//     PAPI support runtime will depend on the runtime that launches the threads
//     instrumented with PAPI.
//
// While we could do a proper topological sort, we take some shortcuts.
// The numerical value of RT_COMMON is guaranteed to be zero. Since all runtimes
// depend on it, it is always the root of the dependence tree. The numerical
// values of the tapir targets are guaranteed to be less than those of PAPI.
// Simply sorting the requested runtimes numerically is, therefore, sufficient
// to get an effective topological sort. The result is a suitable initialization
// order for the runtimes.
//
// If \p reverse is true, the reverse of the dependence order is returned. This
// is a suitable finalization order.
static std::vector<RTID> topoSortRuntimes(const InitOptions &initOpts,
                                          bool reverse = false) {
  std::vector<RTID> ids = {RT_COMMON};
  for (uint64_t id = 0x1, raw = initOpts.rts; raw; raw >>= 1, id <<= 1)
    if (raw & 0x1)
      ids.push_back(static_cast<RTID>(id));
  if (reverse)
    std::reverse(ids.begin(), ids.end());
  return ids;
}

template <typename T, typename... Args> static void add(Args &&...args) {
  gctx.add(new T);
  gctx.get<T>().initialize(args...);
}

#ifdef KITSUNE_PAPI_ENABLED
namespace {
// Trait to check if a class has a member named `getThreadID` that does not
// take any arguments and returns a KitThreadID.
template <typename T> struct matches : std::false_type {};
template <typename R, typename C>
struct matches<R (C::*)() const>
    : std::bool_constant<std::is_same_v<R, KitThreadID>> {};

template <typename T, typename = void>
struct has_getThreadID : std::false_type {};
template <typename T>
struct has_getThreadID<T, std::void_t<decltype(&T::getThreadID)>>
    : std::bool_constant<matches<decltype(&T::getThreadID)>::value> {};
} // namespace

template <typename T> static KitThreadID getThreadID() {
  return gctx.get<T>().getThreadID();
}

template <typename T> static PAPIThreadIDFunc *getThreadIDFunc() {
  if constexpr (std::is_complete_v<T> && has_getThreadID<T>::value)
    return getThreadID<T>;
  else
    return nullptr;
}

static PAPIThreadIDFunc *getThreadIDFunc(RTID rt) {
  switch (rt) {
  case RT_OPENCILK: return getThreadIDFunc<OpenCilkContext>();
  case RT_OPENMP: return getThreadIDFunc<OpenMPContext>();
  case RT_PTHREADS: return getThreadIDFunc<PthreadsContext>();
  case RT_QTHREADS: return getThreadIDFunc<QthreadsContext>();
  case RT_CUDA:
  case RT_HIP:
    // These tapir targets do not use CPU threads. If we return nullptr,
    // papi_thread_init() will not be called.
    return nullptr;
  case RT_COMMON:
  case RT_PAPI:
  case RT_TIMER: UNREACHABLE("Runtime is not tapir-target-specific");
  }
  FATAL("getThreadIDFunc: RTID not handled");
}

template <> void add<PAPIContext>(const InitOptions &initOpts) {
  PAPIThreadIDFunc *threadIDFunc = nullptr;
  for (uint64_t id = 0x1, raw = initOpts.rts; raw && id <= 0xffffffff;
       raw >>= 1, id <<= 1) {
    if (raw & 0x1) {
      if (PAPIThreadIDFunc *func = getThreadIDFunc(static_cast<RTID>(id))) {
        if (!threadIDFunc) {
          threadIDFunc = func;
        } else {
          // If threadIDFunc has already been set, then we have multiple
          // threaded CPU runtimes operating simultaneously.
          FATAL("PAPI does not support multiple threaded CPU runtimes");
        }
      }
    }
  }

  gctx.add(new PAPIContext);
  gctx.get<PAPIContext>().initialize(threadIDFunc);
}
#endif // KITSUNE_PAPI_ENABLED

template <> void add<Context>() {
  // The global singleton context object is guaranteed to exist since it is
  // simply a global variable. There is nothing to be allocated.
  gctx.setVerbose(envLookupOr(envVerbose, envVerboseLegacy, false));
  gctx.setColors(terminalHasColors());
  gctx.setInitialized(true);
}

template <RTID ID, typename... Args> static void initialize(Args &&...args) {
  // We do this first to ensure that RT_COMMON is added to the refcount map in
  // the global singleton runtime context. If the initialization were to fail,
  // the global destructors would be run immediately. These call
  // __kitrt_finalize which would expect RT_COMMON to be present in the map.
  gctx.incr(ID);

  using T = typename context_t<ID>::type;
  if constexpr (!std::is_complete_v<T>) {
    FATAL("Kitsune runtime has not been enabled (%s)", rtname_v<ID>);
  } else if (gctx.initialized<T>()) {
    LOG("Kitsune runtime already initialized (%s)", rtname_v<ID>);
  } else {
    // This is also used to initialize the common parts of Kitsune's runtime. If
    // the runtime has not been initialized, we have to check the environment
    // variables directly to determine if verbose mode has been set.
    LOG_IF_VERBOSE("Initializing Kitsune runtime (%s)", rtname_v<ID>);
    add<T>(args...);
    LOG("Initialized Kitsune runtime (%s)", rtname_v<ID>);
  }
}

template <typename T> static void finalizeImpl() {
  gctx.get<T>().finalize();
  delete gctx.take<T>();
}

template <> void finalizeImpl<Context>() {
  // The global singleton object is a simple global variable. It cannot be
  // deallocated. We could fill it with zero's, but it is not clear what
  // purpose that would serve.
  gctx.setInitialized(false);
}

template <RTID ID> static void finalize() {
  using T = typename context_t<ID>::type;
  if constexpr (!std::is_complete_v<T>) {
    FATAL("Kitsune runtime has not been enabled (%s)", rtname_v<ID>);
  } else if (!gctx.initialized<T>()) {
    LOG("Cannot finalize Kitsune runtime. Not initialized (%s)", rtname_v<ID>);
  } else if (gctx.decr(ID)) {
    LOG("Not finalizing Kitsune runtime. Uses remain (%s)", rtname_v<ID>);
  } else {
    LOG("Finalizing Kitsune runtime (%s)", rtname_v<ID>);
    finalizeImpl<T>();
    // This is also used to finalize the common parts of Kitsune's runtime. If
    // that has been initialized, we have to check the environment variables
    // directly to determine if verbose mode has been set.
    LOG_IF_VERBOSE("Finalized Kitsune runtime (%s)", rtname_v<ID>);
  }
}

static void initialize(const InitOptions &initOpts) {
  auto handle = [](RTID rt, const InitOptions &initOpts) -> void {
    switch (rt) {
    case RT_COMMON: return initialize<RT_COMMON>();
    case RT_CUDA: return initialize<RT_CUDA>();
    case RT_HIP: return initialize<RT_HIP>();
    case RT_OPENCILK: return initialize<RT_OPENCILK>();
    case RT_OPENMP: return initialize<RT_OPENMP>();
    case RT_PAPI: return initialize<RT_PAPI>(initOpts);
    case RT_PTHREADS: return initialize<RT_PTHREADS>();
    case RT_QTHREADS: return initialize<RT_QTHREADS>();
    case RT_TIMER: return initialize<RT_TIMER>();
    }
    FATAL("initialize: RTID not handled");
  };

  for (RTID rt : topoSortRuntimes(initOpts))
    handle(rt, initOpts);
}

static void finalize(const InitOptions &initOpts) {
  auto handle = [](RTID rt) -> void {
    switch (rt) {
    case RT_COMMON: return finalize<RT_COMMON>();
    case RT_CUDA: return finalize<RT_CUDA>();
    case RT_HIP: return finalize<RT_HIP>();
    case RT_OPENCILK: return finalize<RT_OPENCILK>();
    case RT_OPENMP: return finalize<RT_OPENMP>();
    case RT_PAPI: return finalize<RT_PAPI>();
    case RT_PTHREADS: return finalize<RT_PTHREADS>();
    case RT_QTHREADS: return finalize<RT_QTHREADS>();
    case RT_TIMER: return finalize<RT_TIMER>();
    }
    FATAL("initializeRuntimes: RTID not handled");
  };

  for (RTID rt : topoSortRuntimes(initOpts, /*reverse=*/true))
    handle(rt);
}

// ----------------------------------------------------------------------------
// Everything below this is the public interface to the runtime.

extern "C" bool __kitrt_initialized(void) { return gctx.initialized(); }

extern "C" void __kitrt_initialize(const InitOptions *initOpts) {
  assert(initOpts && "Initialization options provided");
  initialize(*initOpts);
}

extern "C" void __kitrt_finalize(const InitOptions *initOpts) {
  assert(initOpts && "Initialization options provided");
  finalize(*initOpts);
}
