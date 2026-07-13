//===- kitomp.cpp - Kitsune runtime targeting LLVM's OpenMP runtime -------===//
//
// Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021, 2023. Los Alamos National Security, LLC. This
//  software was produced under U.S. Government contract
//  DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which
//  is operated by Los Alamos National Security, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT
//  NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
//  OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
//  If software is modified to produce derivative works, such modified
//  software should be clearly marked, so as not to confuse it with
//  the version available from LANL.
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

#include "kitomp.h"
#include "common/env.h"
#include "common/logging.h"
#include "kitrt.h"

// This is an internal header in LLVM's OpenMP runtime. The path is relative
// to ${LLVM_MONOREPO_SOURCE_DIR}/openmp/runtime/src.
#include "kmp.h"

#include <cassert>
#include <string_view>

#define LABEL "kitomp"

using namespace kitrt;

// Since we have included kmp.h, the implementation header for OpenMP, we cannot
// also include omp.h. Therefore, we have to redeclare the functions from omp.h
// that we use.
extern "C" unsigned omp_get_thread_num(void);
extern "C" unsigned omp_get_num_threads(void);
extern "C" unsigned omp_get_max_threads(void);

namespace {

class KitOMPSingleton;

static void newSingleton(void);
static void delSingleton(void);
static KitOMPSingleton *getSingleton(void);

/// Global state for this runtime. We intentionally keep the members public
/// because it is not clear what advantage there is to hiding them.
class KitOMPSingleton {
public:
  // Currently, there are no members. This runtime only needs to know if it has
  // been initialized. If the global singleton is not nullptr, then we know that
  // the runtime has been initialized.

private:
  KitOMPSingleton() = default;
  ~KitOMPSingleton() = default;

public:
  friend void newSingleton(void);
  friend void delSingleton(void);
};

/// FIXME: This should eventually be folded into a single global state object
/// for kitrt - whenever that happens. This will be created by
/// __kitomp_initialize and deleted by __kitomp_finalize. This object should
/// never be accessed directly. Instead, the kitompSingleton*() functions should
/// be used.
static KitOMPSingleton *gSingleton = nullptr;

static void newSingleton(void) { gSingleton = new KitOMPSingleton(); }

static void delSingleton(void) {
  delete gSingleton;
  gSingleton = nullptr;
}

static KitOMPSingleton *getSingleton(void) { return gSingleton; }

} // namespace

/// "Location" information needed by libomp's functions. It would be good to
/// get actual source information, but that would need to come from the
/// frontend, which would be a more complicated implementation. It is not yet
/// clear if this would be useful.
static constexpr const char *unknownLocStr = ";unknown;unknown;0;0;;";
static constexpr int32_t unknownLocSize =
    std::string_view(unknownLocStr).size();

// The fields of the ident_t structure are described below, with comments
// specific to the way they are used by Kitsune.
//
//   struct ident_t {
//      // The documentation says that this might be used in Fortran. But we
//      // don't use it, even in Fortran.
//      int32_t reserved_1;
//
//      // KMP_IDENT_* flags.
//      int32_t flags;
//
//      // reserved_2: The documentation says that this is not used in Fortran
//      // any longer. Since we don't do anything different for Fortran anyway,
//      // it should be ok to justleave this at 0.
//      int32_t reserved_2;
//
//      // reserved_3. The comments in kmp.h say Source[4] in Fortran, do not
//      // use for C++. Naturally, clang is quite happy to use this for both C
//      // and C++. The value is the length of the string describing the source
//      // location in the psource field of this struct.
//      int32_t reserved_3;
//
//      // String describing the source location.
//      const char *psource;
//   };
//

/// Generic location information for the calls to initialize and finalize LLVM's
/// OpenMP runtime.
static ident_t unknownLoc = {
    0, KMP_IDENT_KMPC, 0, unknownLocSize, unknownLocStr,
};

/// Generic location information for a standard parallel for loop.
static ident_t staticLoopLoc = {
    0, KMP_IDENT_KMPC | KMP_IDENT_WORK_LOOP, 0, unknownLocSize, unknownLocStr,
};

/// Get the number of threads available for parallel execution. For consistency,
/// this should be used instead of directly calling omp_get_max_threads().
extern "C" uint64_t __kitomp_num_threads(void) {
  assert(__kitomp_initialized() && "kitomp initialized");
  return omp_get_max_threads();
}

/// The number of partial reductions to perform in parallel.
///
/// \param n The trip count of the parallel loop containing a reduction
extern "C" uint64_t __kitomp_reduce_num_partials(uint64_t n) {
  assert(__kitomp_initialized() && "kitomp initialized");

  log(LABEL, "Calculating number of partial reductions");

  // There might be something smarter that can be done once we support a proper
  // reduction tree, but since we only support a reduction tree of depth 1, we
  // just return the number of CPU's on the system.
  uint64_t numPartials = __kitomp_num_threads();

  log(LABEL, "Number of partial reductions: %d", numPartials);

  return numPartials;
}

/// This wraps the function \p f that will be launched on each thread. It
/// calculates the range of iterations that should be executed by \p f, then,
/// invokes it with that range. This uses static scheduling - so every thread
/// will execute a roughly equal subset of the iterations.
///
/// The exact subset of the iterations to calculate on a thread is determined
/// by libomp functions that are called here.
static void staticLoopWrapper(int32_t *globalTID, int32_t *localTID,
                              KitOMPThrdFunc f, uint64_t start, uint64_t end,
                              void *args) {
  log(LABEL, "Running on thread %d of %d [global = %d]", *localTID,
      omp_get_num_threads(), *globalTID);

  // Since each thread must run exactly one iteration, we can just use the
  // thread id to specify the range.
  f(*localTID, *localTID + 1, args);
}

/// Launch some number of threads each of which will execute some number of
/// iterations in the space [\p start, \p end). This blocks until all threads
/// have completed. The compiler will transform all tapir loops so they are of
/// the following form:
///
///     unsigned numThreads = __kitomp_num_threads();
///     size_t itersPerThread = (numThreads + n - 1) / numThreads
///     forall (unsigned t = 0; t < numThrds; ++t) {
///       size_t start = t * itersPerThread;
///       size_t end = std::min(start + itersPerThread, n);
///       for (size_t i = start; i < end; ++i)
///         ...
///     }
///
/// This function, therefore, will launch exactly `end - start` threads, each of
/// which will execute exactly one iteration. In the future, `end - start` may
/// be less than the number of threads available.
///
/// \param f The function to execute on each thread
/// \param start The start index of the iteration space
/// \param end The value one greater than the last index of the iteration space
/// \param args Pointer to the struct containing data to be passed to \p f
extern "C" void __kitomp_launch(KitOMPThrdFunc f, uint64_t start, uint64_t end,
                                void *args) {
  assert(__kitomp_initialized() && "kitomp initialized");
  assert(start == 0 && end == __kitomp_num_threads() &&
         "__kitomp_launch expects loop iterations in range [0,NUM_THREADS)");
  log(LABEL, "Launching multithreaded loop: [%ld,%ld)", start, end);

  // This will launch `omp_get_max_threads()` threads.
  __kmpc_fork_call(&staticLoopLoc, 4, (kmpc_micro)&staticLoopWrapper, f, start,
                   end, args);

  log(LABEL, "Finished multithreaded loop");
}

/// Check if this runtime has already been initialized.
extern "C" bool __kitomp_initialized(void) { return getSingleton(); }

/// Get a thread ID suitable for use in PAPI.
static unsigned long getThreadIDForPAPI(void) { return omp_get_thread_num(); }

/// Initialize kitsune's OpenMP runtime as well as the actual OpenMP runtime.
extern "C" void __kitomp_initialize(void) {
  if (__kitomp_initialized()) {
    log(LABEL, "Runtime already initialized");
    return;
  }

  logEarly(LABEL, "Initializing Kitsune runtime (openmp)");

  // Create the global singleton object.
  newSingleton();

  // Initialize the components of kitsune's runtime that are shared by the
  // tapir-target-specific components.
  __kitrt_initialize();

#ifdef KITRT_PAPI_ENABLED
  __kitpapi_initialize(getThreadIDForPAPI);
#endif // KITRT_PAPI_ENABLED

  uint64_t numThreads = __kitrt_num_threads("OMP_NUM_THREADS");
  envSet("OMP_NUM_THREADS", numThreads);

  // The second argument in the call to __kmpc_begin is currently unused, per
  // the 10-year old documentation that seems to be the only kind that is
  // available.
  log(LABEL, "Initializing OpenMP runtime");
  __kmpc_begin(&unknownLoc, /*flags=*/0);
  log(LABEL, "Initialized OpenMP runtime");

  log(LABEL, "Number of threads = %d", __kitomp_num_threads());
  log(LABEL, "Initialized Kitsune runtime (openmp)");
}

/// Finalize kitsune's OpenMP runtime, as well as OpenMP runtime.
extern "C" void __kitomp_finalize(void) {
  if (!__kitomp_initialized()) {
    log(LABEL, "Cannot finalize runtime. Not initialized");
    return;
  }

  log(LABEL, "Finalizing Kitsune runtime (openmp)");

  // This call is optional, but we use it anyway for consistency with the other
  // runtimes.
  log(LABEL, "Finalizing OpenMP runtime");
  __kmpc_end(&unknownLoc);
  log(LABEL, "Finalized OpenMP runtime");

  // Finalize the components of Kitsune's runtime that are shared by the
  // tapir-target-specific components.
  __kitrt_finalize();

  // Delete the global singleton object.
  delSingleton();

  log(LABEL, "Finalized Kitsune runtime (openmp)");
}
