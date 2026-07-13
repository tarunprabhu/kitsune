//===- kitpthr.cpp - Runtime for Kitsune's pthreads tapir target ----------===//
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
//
// This targets the POSIX threads (pthreads) library.
//
//===----------------------------------------------------------------------===//

#include "kitpthr.h"
#include "common/env.h"
#include "common/logging.h"
#include "kitrt.h"

#include <cassert>
#include <errno.h>
#include <pthread.h>
#include <vector>

#define LABEL "kitpthr"

using namespace kitrt;

/// The type of the pthread start function.
using PthreadStartFunc = void *(*)(void *);

namespace {

/// Metadata for each thread. This is passed to the thread launch function and
/// also contains the arguments to be passed to the "actual" thread function.
struct KitPthrThread {
  KitPthrThrdFunc f;
  int64_t tid;
  void *args;
  pthread_t pthr;
  pthread_attr_t attr;
};

/// The thread context object. This is created and returned by
/// \ref __kitpthr_launch. This should be passed to __kitpthr_join where it
/// will be deleted.
struct KitPthrContextImpl {
  std::vector<KitPthrThread> thrds;

public:
  KitPthrContextImpl() = default;
  KitPthrContextImpl(size_t numThreads) : thrds(numThreads) {
    for (size_t t = 0; t < numThreads; ++t)
      thrds[t].tid = t;
  }

  KitPthrContextImpl(const KitPthrContextImpl &) = delete;
  KitPthrContextImpl(KitPthrContextImpl &&) = delete;
  KitPthrContext &operator=(const KitPthrContextImpl &) = delete;

  KitPthrThread &operator[](size_t i) { return thrds.at(i); }
  const KitPthrThread &operator[](size_t i) const { return thrds.at(i); }
  size_t size() const { return thrds.size(); }

  decltype(thrds)::iterator begin() { return thrds.begin(); }
  decltype(thrds)::const_iterator begin() const { return thrds.begin(); }
  decltype(thrds)::iterator end() { return thrds.end(); }
  decltype(thrds)::const_iterator end() const { return thrds.end(); }
};

} // namespace

[[noreturn]] static void kitpthrHandleCreateError(int err) {
  const char *lede = "Could not create thread";
  switch (err) {
  case EINVAL:
    fatal(LABEL, "%s. Invalid attributes", lede);
  case EAGAIN:
    fatal(LABEL, "%s. Insufficient resources", lede);
  case EPERM:
    fatal(LABEL, "%s. Insufficient permissions", lede);
  default:
    fatal(LABEL, "%s. Unknown error", lede);
  }
}

[[noreturn]] static void kitpthrHandleJoinError(int err) {
  const char *lede = "Error joining thread";
  switch (err) {
  case EDEADLK:
    fatal(LABEL, "%s. Deadlock detected", lede);
  case EINVAL:
    fatal(LABEL, "%s. Thread is not joinable", lede);
  case ESRCH:
    fatal(LABEL, "%s. Invalid thread id", lede);
  default:
    fatal(LABEL, "%s. Unknown error", lede);
  }
}

/// The number of threads to use. Ideally, this should be part of a global
/// object that contains all the state needed by the runtime. But that would
/// require reorganization of the runtime. A separate effort is underway that
/// does this, so do this temporarily for now.
///
/// This will be set in the global ctor for this runtime. This should not be
/// used directly in the rest of the runtime. Use `__kitpthr_num_threads()` to
/// get this value.
static uint64_t __kitpthr_num_threads_v = 1;

/// Get the number of threads available for parallel work.
extern "C" uint64_t __kitpthr_num_threads(void) {
  return __kitpthr_num_threads_v;
}

/// The number of partial reductions to perform in parallel.
///
/// \param n The trip count of the parallel loop containing a reduction
extern "C" uint64_t __kitpthr_reduce_num_partials(uint64_t n) {
  log(LABEL, "Calculating number of partial reductions");

  // There might be something smarter that can be done once we support a proper
  // reduction tree, but since we only support a reduction tree of depth 1, we
  // just use as many partials as there are CPU's on the system.
  uint64_t numPartials = __kitpthr_num_threads();

  log(LABEL, "Number of partial reductions: %d", numPartials);

  return numPartials;
}

/// The function that is launched by each thread. This simply finds the "actual"
/// function that is to be run in \p thrdInfo and calls it. The arguments to the
/// actual function are also present in \p thrdInfo. Always returns 0.
static void *kitpthrThrdStartFn(KitPthrThread *thread) {
  KitPthrThrdFunc f = thread->f;
  int64_t tid = thread->tid;
  void *args = thread->args;

  f(tid, tid + 1, args);

  return nullptr;
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
/// This function, therefore, will launch exactly `end - start - 1` threads,
/// each of which will execute exactly one iteration. The main thread will
/// execute the remaining iteration. It will, therefore, block until that
/// iteration has completed. In the future, `end - start` may be less than the
/// number of threads available.
///
/// \param f The function to execute on each thread
/// \param start The start index of the iteration space
/// \param end The value one greater than the last index of the iteration space
/// \param args A struct containing data to be passed to \p f
/// \return An opaque thread context object. It is the caller's responsibility
/// to call \ref __kitpthr_join with this context object. If no threads are
/// launched, i.e. \p f is run on the main thread, nullptr will be returned
/// instead. In this case, the caller is not required to call
/// \ref __kitpthr_join.
extern "C" KitPthrContext *__kitpthr_launch(KitPthrThrdFunc f, uint64_t start,
                                            uint64_t end, void *args) {
  assert(start == 0 && end == __kitpthr_num_threads() &&
         "__kitpthr_launch expects loop iterations in range [0,NUM_THREADS)");
  log(LABEL, "Launching multithreaded loop: [%ld,%ld)", start, end);

  uint64_t numThreads = __kitpthr_num_threads();
  KitPthrContextImpl *ctx = new KitPthrContextImpl(numThreads - 1);
  for (KitPthrThread &thrd : *ctx) {
    thrd.f = f;
    thrd.args = args;

    if (pthread_attr_init(&thrd.attr))
      fatal(LABEL, "Error initializing thread attributes");

    if (int err = pthread_create(&thrd.pthr, &thrd.attr,
                                 (PthreadStartFunc)kitpthrThrdStartFn, &thrd))
      kitpthrHandleCreateError(err);
    log(LABEL, "Fork thread %ld (%ld)", thrd.tid, thrd.pthr);
  }
  f(numThreads - 1, numThreads, args);

  return reinterpret_cast<KitPthrContext *>(ctx);
}

/// Join the threads launched by a previous call to \ref __kitpthr_launch.
/// \p ctx is the context returned by that call. \p ctx may be nullptr, in which
/// case, this function does nothing.
extern "C" void __kitpthr_sync(KitPthrContext *p) {
  KitPthrContextImpl *ctx = reinterpret_cast<KitPthrContextImpl *>(p);
  log(LABEL, "Joining %ld threads", ctx->size());
  for (KitPthrThread &thrd : *ctx) {
    if (int err = pthread_join(thrd.pthr, nullptr))
      kitpthrHandleJoinError(err);
    if (pthread_attr_destroy(&thrd.attr))
      fatal(LABEL, "Error destroying thread attributes");
    log(LABEL, "Joined thread %ld (%ld)", thrd.tid, thrd.pthr);
  }
  delete ctx;
}

/// Get a thread ID suitable for use by PAPI.
static unsigned long getThreadIDForPAPI(void) { return pthread_self(); }

/// Initialize kitsune's pthreads runtime. Currently, this only sets some
/// global variables that enable verbose mode. This runtime is not intended to
/// ever maintain any other state.
extern "C" void __kitpthr_initialize(void) {
  logEarly(LABEL, "Initializing Kitsune runtime (pthreads)");

  // Initialize the components of kitsune's runtime that are shared by the
  // tapir-target-specific components.
  __kitrt_initialize();

#ifdef KITRT_PAPI_ENABLED
  __kitpapi_initialize(getThreadIDForPAPI);
#endif // KITRT_PAPI_ENABLED

  __kitpthr_num_threads_v = __kitrt_num_threads(nullptr);

  // pthreads does not have to be initialized.

  log(LABEL, "Number of threads = %d", __kitpthr_num_threads());
  log(LABEL, "Initialized Kitsune runtime (pthreads)");
}

/// Finalize kitsune's pthreads runtime. This is only present for symmetry with
/// \ref __kitpthr_initialize. Since the runtime does not maintain any global
/// state of its own, this does nothing.
extern "C" void __kitpthr_finalize(void) {
  log(LABEL, "Finalizing Kitsune runtime (pthreads)");

  // pthreads does not need to be finalized.

  // Finalize the components of Kitsune's runtime that are shared by the
  // tapir-target-specific components.
  __kitrt_finalize();

  log(LABEL, "Finalized Kitsune runtime (pthreads)");
}
