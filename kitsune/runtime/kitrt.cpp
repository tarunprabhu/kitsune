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
#include "common/traits.h"
#include "common/unreachable.h"
#include "global/global.h"
#include "openmp/kitomp.h"
#include "pthreads/kitpthr.h"
#include "timer/timer.h"

#ifdef KITSUNE_COLORS_ENABLED
#include <curses.h>
#include <unistd.h>
#endif // KITSUNE_COLORS_ENABLED

#ifdef KITSUNE_CUDA_ENABLED
#include "cuda/kitcuda.h"
#endif // KITSUNE_CUDA_ENABLED

#ifdef KITSUNE_HIP_ENABLED
#include "hip/kithip.h"
#endif // KITSUNE_HIP_ENABLED

#ifdef KITSUNE_OPENCILK_ENABLED
#include "opencilk/kitocilk.h"
#endif // KITSUNE_OPENCILK_ENABLED

#ifdef KITSUNE_PAPI_ENABLED
#include "papi/kitpapi.h"
#endif // KITSUNE_PAPI_ENABLED

#ifdef KITSUNE_QTHREADS_ENABLED
#include "qthreads/kitqthr.h"
#endif // KITSUNE_QTHREADS_ENABLED

#include <algorithm>
#include <vector>

using namespace kitrt;

static bool terminalHasColors() {
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
  bool colors = has_colors();
  endwin();
  delscreen(scr);

  return colors;
#else  // !KITSUNE_COLORS_ENABLED
  return false;
#endif // !KITSUNE_COLORS_ENABLED
}

static std::vector<RTID> getRuntimes(uint64_t raw, uint64_t id0) {
  std::vector<RTID> ids;
  for (uint64_t id = id0; raw; raw >>= 1, id <<= 1)
    if (raw & 0x1)
      ids.push_back(static_cast<RTID>(id));
  return ids;
}

static std::vector<RTID> getTTRuntimes(const InitOptions &initOpts) {
  return getRuntimes(initOpts.rts & 0xffffffffULL, 0x1);
}

static std::vector<RTID> getInstrRuntimes(const InitOptions &initOpts) {
  return getRuntimes(initOpts.rts >> 32, 0x100000000ULL);
}

template <typename T, typename... Args> static void initialize(Args &&...args) {
  if constexpr (!std::is_complete_v<T>) {
    FATAL("Kitsune runtime has not been enabled (%s)", T::name());
  } else if (gctx.has<T>()) {
    LOG("Kitsune runtime already initialized (%s)", T::name());
  } else {
    LOG("Initializing Kitsune runtime (%s)", T::name());
    mutCtx().addContext(new T);
    mutCtx<T>().initialize(args...);
    LOG("Initialized Kitsune runtime (%s)", T::name());
  }
}

static void initializePAPI(const InitOptions &initOpts) {
  if constexpr (std::is_complete_v<KitPAPIContext>) {
    auto getThreadIDFunc = [](RTID rt) -> PAPIThreadIDFunc * {
      switch (rt) {
      case RT_OPENCILK: return __kitocilk_worker_id;
      case RT_OPENMP: return __kitomp_thread_id;
      case RT_PTHREADS: return __kitpthr_thread_id;
      case RT_QTHREADS: return __kitqthr_worker_id;
      case RT_CUDA:
      case RT_HIP: return nullptr;
      case RT_PAPI:
      case RT_TIMER: UNREACHABLE("Runtime is not tapir-target-specific");
      }
      FATAL("getThreadIDFunc: RTID not handled");
    };

    std::vector<PAPIThreadIDFunc *> getThreadIDFuncs;
    for (RTID rt : getTTRuntimes(initOpts))
      if (std::optional<PAPIThreadIDFunc *> threadIDFunc = getThreadIDFunc(rt))
        getThreadIDFuncs.push_back(*threadIDFunc);

    if (getThreadIDFuncs.size() > 1)
      WARN("PAPI not initialized. Initialized more than one runtime supporting "
           "PAPI");
    else if (getThreadIDFuncs.size() < 1)
      initialize<KitPAPIContext>(nullptr);
    else
      initialize<KitPAPIContext>(getThreadIDFuncs[0]);
  } else {
    FATAL("Kitsune runtime has not been enabled (%s)", KitPAPIContext::name());
  }
}

static void initialize(const InitOptions &initOpts, RTID rt) {
  switch (rt) {
  case RT_CUDA: return initialize<KitCudaContext>();
  case RT_HIP: return initialize<KitHipContext>();
  case RT_OPENCILK: return initialize<KitOCilkContext>();
  case RT_OPENMP: return initialize<KitOMPContext>();
  case RT_PAPI: return initializePAPI(initOpts);
  case RT_PTHREADS: return initialize<KitPthrContext>();
  case RT_QTHREADS: return initialize<KitQthrContext>();
  case RT_TIMER: return initialize<KitTimerContext>();
  }
  FATAL("initializeRuntimes: RTID not handled");
}

// Initialize the requested tapir-target-specific runtimes, as well as any
// support runtimes.
static void initializeRuntimes(const InitOptions &initOpts) {
  // Initialize the tapir-target-specific runtimes fist. Some instrumentation
  // runtimes, specifically PAPI, require the former to be initialized first.
  for (RTID rt : getTTRuntimes(initOpts))
    initialize(initOpts, rt);

  for (RTID rt : getInstrRuntimes(initOpts))
    initialize(initOpts, rt);
}

// This initializes the common parts of Kitsune's runtime.
static void initializeCommonRuntime(const InitOptions &initOpts) {
  // At the point, the global singleton object has not yet been initialized.
  // As a result, verbose mode will always return false. The only way to ensure
  // that a message is printed is to check the environment variables.
  LOG_IF_VERBOSE("Initializing Kitsune runtime (common)");

  KitRTContext &rt = mutCtx();
  rt.setVerbose(envLookupOr(envVerbose, envVerboseLegacy, false));
  rt.setColors(terminalHasColors());

  rt.setInitialized(true);
  LOG("Initialized Kitsune runtime (common)");
}

template <typename T> static void finalize() {
  if constexpr (std::is_complete_v<T>) {
    if (gctx.has<T>()) {
      LOG("Finalizing Kitsune runtime (%s)", T::name());
      mutCtx<T>().finalize();
      delete mutCtx().takeContext<T>();
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
  case RT_CUDA: return finalize<KitCudaContext>();
  case RT_HIP: return finalize<KitHipContext>();
  case RT_OPENCILK: return finalize<KitOCilkContext>();
  case RT_OPENMP: return finalize<KitOMPContext>();
  case RT_PAPI: return finalize<KitPAPIContext>();
  case RT_PTHREADS: return finalize<KitPthrContext>();
  case RT_QTHREADS: return finalize<KitQthrContext>();
  case RT_TIMER: return finalize<KitTimerContext>();
  }
  FATAL("initializeRuntimes: RTID not handled");
}

static void finalizeRuntimes(const InitOptions &initOpts) {
  // In general, we should be finalizing runtimes in the opposite order in
  // which they were initialized. In particular, the PAPI runtime must be
  // finalized after the tapir-target-specific runtimes since it may use the
  // thread id functions from the former.
  for (RTID rt : getInstrRuntimes(initOpts))
    finalize(rt);

  for (RTID rt : getTTRuntimes(initOpts))
    finalize(rt);
}

static void finalizeCommonRuntime() {
  LOG("Finalizing Kitsune runtime (common)");

  mutCtx().setInitialized(false);

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
