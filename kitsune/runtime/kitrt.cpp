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
#include "global/global.h"
#include "openmp/kitomp.h"
#include "pthreads/kitpthr.h"
#include "timer/timer.h"

#ifdef KITRT_COLORS_ENABLED
#include <curses.h>
#include <unistd.h>
#endif // KITRT_COLORS_ENABLED

#ifdef KITRT_CUDA_ENABLED
#include "cuda/kitcuda.h"
#endif // KITRT_CUDA_ENABLED

#ifdef KITRT_HIP_ENABLED
#include "hip/kithip.h"
#endif // KITRT_HIP_ENABLED

#ifdef KITRT_OPENCILK_ENABLED
#include "opencilk/kitocilk.h"
#endif // KITRT_OPENCILK_ENABLED

#ifdef KITRT_PAPI_ENABLED
#include "papi/kitpapi.h"
#endif // KITRT_PAPI_ENABLED

#ifdef KITRT_QTHREADS_ENABLED
#include "qthreads/kitqthr.h"
#endif // KITRT_QTHREADS_ENABLED

#include <algorithm>
#include <vector>

using namespace kitrt;

static bool terminalHasColors() {
  bool colors = false;

#ifdef KITRT_COLORS_ENABLED
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

  // Otherwise, do the "sane" thing and use colors only if the terminal supports
  // it. If stderr is not connected to a terminal, don't use colors.
  if (isatty(STDERR_FILENO)) {
    SCREEN *scr = newterm(NULL, stderr, stdin);
    colors = has_colors();
    endwin();
    delscreen(scr);
  }
#endif // KITRT_COLORS_ENABLED

  return colors;
}

static std::vector<RTID> getRTIDs(const InitOptions &initOpts) {
  std::vector<RTID> ids;
  for (uint64_t rts = initOpts.rts, id = 0x1; rts; rts >>= 1, id <<= 1)
    if (rts & 0x1)
      ids.push_back(static_cast<RTID>(id));
  return ids;
}

template <typename T, typename = void> struct is_complete : std::false_type {};
template <typename T>
struct is_complete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};

template <typename T, typename... Args> static void initialize(Args &&...args) {
  if constexpr (is_complete<T>::value) {
    if (!T::initialized()) {
      LOG("Initializing Kitsune runtime (%s)", T::name());
      mutKitRTContext().addContext(new T);
      T::mut().initialize(args...);
      LOG("Initialized Kitsune runtime (%s)", T::name());
    } else {
      LOG("Kitsune runtime already initialized (%s)", T::name());
    }
  } else {
    FATAL("Kitsune runtime has not been enabled (%s)", T::name());
  }
}

static void initialize(RTID rt) {
  switch (rt) {
  case RT_SERIAL:
    // These do not have any associated runtimes that need to be initialized.
    return;
  case RT_CUDA: return initialize<KitCudaContext>();
  case RT_HIP: return initialize<KitHipContext>();
  case RT_OPENCILK: return initialize<KitOCilkContext>();
  case RT_OPENMP: return initialize<KitOMPContext>();
  case RT_PTHREADS: return initialize<KitPthrContext>();
  case RT_QTHREADS: return initialize<KitQthrContext>();
  case RT_TIMER: return initialize<KitTimerContext>();
  case RT_NONE:
  case RT_PAPI:
    // This should never happen.
    break;
  }
  FATAL("initializeRuntimes: RTID not handled");
}

static void initializePAPI(const InitOptions &initOpts) {
  if constexpr (is_complete<KitPAPIContext>::value) {
    auto getThreadIDFunc = [](RTID rt) -> std::optional<PAPIThreadIDFunc *> {
      switch (rt) {
      case RT_OPENCILK: return __kitocilk_worker_id;
      case RT_OPENMP: return __kitomp_thread_id;
      case RT_PTHREADS: return __kitpthr_thread_id;
      case RT_QTHREADS: return __kitqthr_worker_id;
      case RT_SERIAL: return nullptr;
      default: return std::nullopt;
      }
    };

    std::vector<PAPIThreadIDFunc *> getThreadIDFuncs;
    for (RTID rt : getRTIDs(initOpts))
      if (std::optional<PAPIThreadIDFunc *> threadIDFunc = getThreadIDFunc(rt))
        getThreadIDFuncs.push_back(*threadIDFunc);

    if (getThreadIDFuncs.empty())
      WARN("PAPI not initialized. Initialized zero runtimes supporting PAPI");
    else if (getThreadIDFuncs.size() > 1)
      WARN("PAPI not initialized. Initialized more than one runtime supporting "
           "PAPI");
    else
      initialize<KitPAPIContext>(getThreadIDFuncs[0]);
  } else {
    FATAL("Kitsune runtime has not been enabled (%s)", KitPAPIContext::name());
  }
}

// Initialize the requested tapir-target-specific runtimes, as well as any
// support runtimes.
static void initializeRuntimes(const InitOptions &initOpts) {
  // Initializing PAPI is more involved, so initialize everything else first.
  for (RTID rt : getRTIDs(initOpts))
    if (rt != RT_PAPI)
      initialize(rt);

  if (initOpts.rts & RT_PAPI)
    initializePAPI(initOpts);
}

// This initializes the common parts of Kitsune's runtime.
static void initializeCommonRuntime(const InitOptions &initOpts) {
  // At the point, the global singleton object has not yet been initialized.
  // As a result, verbose mode will always return false. The only way to ensure
  // that a message is printed is to check the environment variables.
  LOG_IF_VERBOSE("Initializing Kitsune runtime (common)");

  KitRTContext &rt = mutKitRTContext();
  rt.setVerbose(envLookupOr(envVerbose, envVerboseLegacy, false));
  rt.setColors(terminalHasColors());

  rt.setInitialized(true);
  LOG("Initialized Kitsune runtime (common)");
}

template <typename T> static void finalize() {
  if constexpr (is_complete<T>::value) {
    if (T::initialized()) {
      LOG("Finalizing Kitsune runtime (%s)", T::name());
      T::mut().finalize();
      delete mutKitRTContext().takeContext<T>();
      LOG("Finalized Kitsune runtime (%s)", T::name());
    } else {
      LOG("Cannot finalize Kitsune runtime. Not initialized (%s)", T::name());
    }
  } else {
    FATAL("Kitsune runtime has not been enabled (%s)", T::name());
  }
}

static void finalize(RTID rt) {
  switch (rt) {
  case RT_SERIAL:
    // These do not have any associated runtimes that need to be initialized.
    return;
  case RT_CUDA: return finalize<KitCudaContext>();
  case RT_HIP: return finalize<KitHipContext>();
  case RT_OPENCILK: return finalize<KitOCilkContext>();
  case RT_OPENMP: return finalize<KitOMPContext>();
  case RT_PAPI: return finalize<KitPAPIContext>();
  case RT_PTHREADS: return finalize<KitPthrContext>();
  case RT_QTHREADS: return finalize<KitQthrContext>();
  case RT_TIMER: return finalize<KitTimerContext>();
  case RT_NONE:
    // This should never happen.
    break;
  }
  FATAL("initializeRuntimes: RTID not handled");
}

static void finalizeRuntimes(const InitOptions &initOpts) {
  // In general, we should be finalizing runtimes in the opposite order in
  // which they were initialized. In particular, the PAPI runtime must be
  // finalized last since it may use the thread id functions from the
  // tapir-target-specific runtimes. The ID of the former is guaranteed to be
  // numerically larger than the latter, simply reversing the list of runtime
  // ID's is sufficient.
  std::vector<RTID> rts = getRTIDs(initOpts);
  std::reverse(rts.begin(), rts.end());
  for (RTID rt : rts)
    finalize(rt);
}

static void finalizeCommonRuntime() {
  LOG("Finalizing Kitsune runtime (common)");

  mutKitRTContext().setInitialized(false);

  // Although we don't do so, it is reasonable to clear the global singleton at
  // this point. In that case, LOG(...) would not work because verbose mode
  // would always be false. Using LOG_IF_VERBOSE(...) here serves as a marker
  // that the global context object should not be used beyond this point.
  LOG_IF_VERBOSE("Finalized Kitsune runtime (common)");
}

extern "C" bool __kitrt_initialized(void) { return gctx.initialized; }

extern "C" void __kitrt_initialize(const InitOptions *initOpts) {
  assert(initOpts && "Initialization options provided");

  if (!__kitrt_initialized()) {
    initializeCommonRuntime(*initOpts);
    initializeRuntimes(*initOpts);
  } else {
    LOG("Kitsune runtime already initialized (common)");

    // Even if the runtime has already been initialized, we may have to
    // initialize additional runtimes - either for specific tapir targets, or
    // one or more of the supporting runtimes, such as for the timers. In
    // general, we expect __kitrt_initialize to be called once from the global
    // constructor of an executable. But this can also be executed from the
    // global constructor of a dynamic shared object when it is loaded. This may
    // happen after the main ctor has already run. There is no way to know if
    // this will happen ahead of time.
    initializeRuntimes(*initOpts);
  }
}

extern "C" void __kitrt_finalize(const InitOptions *initOpts) {
  assert(initOpts && "Initialization options provided");
  if (__kitrt_initialized()) {
    finalizeRuntimes(*initOpts);
    finalizeCommonRuntime();
  } else {
    LOG("Cannot finalize Kitsune runtime. Not initialized (common)");

    // Even if the runtime has already been finalized, we may have to finalize
    // additional runtimes.
    //
    // FIXME: THIS IS WRONG! The real issue here is that we could have multiple
    // calls to initialize - which must have corresponding calls to finalize.
    // Each such call my initialize a different number of supporting runtimes.
    // In such cases, we should only run the true finalization when we know that
    // it is the last finalization call.
    finalizeRuntimes(*initOpts);
  }
}
