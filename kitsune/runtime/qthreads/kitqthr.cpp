//===- kitqthr.cpp - Runtime for Kitsune's qthreads tapir target ----------===//
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
// This targets Qthreads.
//
//===----------------------------------------------------------------------===//

#include "kitqthr.h"
#include "common/env.h"
#include "common/logging.h"
#include "common/utils.h"
#include "global/singleton.h"
#include "kitrt.h"

#include <qthread.h>
#include <qthread/barrier.h>

#include <cassert>
#include <vector>

using namespace kitrt;

namespace kitrt {

/// Global state for this runtime. We intentionally keep the members public
/// because it is not clear what advantage there is to hiding them.
class KitQthrContext : public KitContextMixin<KitQthrContext> {
public:
  // Currently, there are no members. This runtime only needs to know if it has
  // been initialized. If the global singleton is not nullptr, then we know that
  // the runtime has been initialized.
};

} // namespace kitrt

namespace {

/// The arguments is passed to the thread launch function. This contains the
/// arguments to be passed to the "actual" thread function. It also contains the
/// barrier that must be entered once the actual thread function has finished
/// executing.
struct KitQthrThrdArgs {
  KitQthrThrdFunc f;
  int64_t tid;
  void *args;
  qt_barrier_t *barrier;
};

} // namespace

/// Get the number of parallel workers that are available. Generally, this
/// function should be used when this must be queried instead of calling
/// `qthread_num_workers()`.
extern "C" uint64_t __kitqthr_num_workers(void) {
  assert(__kitqthr_initialized() && "kitqthr initialized");
  return qthread_num_workers();
}

/// Get the ID of the worker from which this is called.
extern "C" uint64_t __kitqthr_worker_id(void) {
  return qthread_worker(nullptr);
}

/// The number of partial reductions to perform in parallel.
///
/// \param n The trip count of the parallel loop containing a reduction
extern "C" uint64_t __kitqthr_reduce_num_partials(uint64_t n) {
  assert(__kitqthr_initialized() && "kitqthr initialized");
  LOG("Calculating number of partial reductions");

  // There might be something smarter that can be done once we support a proper
  // reduction tree, but since we only support a reduction tree of depth 1, this
  // will do.
  uint64_t numPartials = __kitqthr_num_workers();

  LOG("Number of partial reductions: %d", numPartials);

  return numPartials;
}

/// The function that is launched by each thread. This simply finds the "actual"
/// function that is to be run in \p thrdArgs and calls it. The arguments to the
/// actual function are also present in \p thrdArgs. Always returns 0.
static unsigned long kitqthrThrdLaunchFn(KitQthrThrdArgs *thrdArgs) {
  KitQthrThrdFunc f = thrdArgs->f;
  int64_t tid = thrdArgs->tid;
  void *args = thrdArgs->args;
  qt_barrier_t *barrier = thrdArgs->barrier;

  f(tid, tid + 1, args);

  LOG("Thread entering barrier: %d", tid);
  qt_barrier_enter(barrier);

  return 0;
}

/// Launch some number of threads each of which will execute some number of
/// iterations in the space [\p start, \p end). This blocks until all threads
/// have completed. The compiler will transform all tapir loops so they are of
/// the following form:
///
///     unsigned numThreads = __kitqthr_num_threads();
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
/// NOTE: At this time, \p argSize is not used because this function blocks
/// until all threads have finished executing. In the future, if we change this
/// to be non-blocking, \p args will be copied before this returns, at which
/// point, \p argSize will be used.
///
/// \param f The function to execute on each thread
/// \param start The start index of the iteration space
/// \param end The value one greater than the last index of the iteration space
/// \param args A struct containing data to be passed to \p f
/// \param argSize The size of the struct pointed to by \p args.
extern "C" void __kitqthr_launch(KitQthrThrdFunc f, uint64_t start,
                                 uint64_t end, void *args,
                                 [[maybe_unused]] uint64_t argSize) {
  assert(__kitqthr_initialized() && "kitqthr initialized");
  assert(start == 0 && end == __kitqthr_num_workers() &&
         "__kitqthr_launch expects loop iterations in range [0,NUM_THREADS)");
  LOG("Launching multithreaded loop: [%ld,%ld)", start, end);

  uint64_t numThrds = __kitqthr_num_workers();

  // If only a single worker is available, take the quick way out.
  if (numThrds == 1) {
    f(0, 1, args);
    return;
  }

  // We need the main thread to block until all spawned threads finish. This is
  // implemented by setting up the barrier to block until it is entered by all
  // `n` spawned threads, as well as the main thread.
  qt_barrier_t *barrier = qt_barrier_create(numThrds + 1, REGION_BARRIER);
  if (!barrier)
    FATAL("Could not create barrier");

  std::vector<KitQthrThrdArgs> thrds(numThrds);
  for (uint64_t t = 0; t < numThrds; ++t) {
    KitQthrThrdArgs &thrdArgs = thrds[t];
    thrdArgs.f = f;
    thrdArgs.tid = t;
    thrdArgs.args = args;
    thrdArgs.barrier = barrier;

    LOG("Fork thread %d", t);
    if (qthread_fork((qthread_f)kitqthrThrdLaunchFn, &thrdArgs, nullptr))
      FATAL("Could not fork thread");
  }

  // The main thread must also enter the barrier. Once the main thread has
  // entered, it will block until all spawned threads enter before continuing.
  qt_barrier_enter(barrier);
  qt_barrier_destroy(barrier);

  LOG("Finished multithreaded loop");
}

/// Check if this runtime has already been initialized.
extern "C" bool __kitqthr_initialized(void) {
  return KitQthrContext::hasSingleton();
}

/// Initialize kitsune's qthreads runtime as well as the underlying Qthreads
/// runtime. This is intended to be called from a global constructor that is
/// generated by Kitsune. This is not thread-safe, but it is safe to call more
/// than once (subsequent calls will return immediately).
extern "C" void __kitqthr_initialize(void) {
  if (__kitqthr_initialized()) {
    LOG("Runtime already initialized");
    return;
  }

  __kitrt_initialize();

  LOG("Initializing Kitsune runtime (qthreads)");

  KitQthrContext::addSingleton();

#ifdef KITRT_PAPI_ENABLED
  __kitpapi_initialize(__kitqthr_worker_id);
#endif // KITRT_PAPI_ENABLED

  uint64_t numThreads = getNumThreadsOrCPUs();
  envSet("QT_NUM_SHEPHERDS", numThreads);
  envSet("QT_NUM_WORKERS_PER_SHEPHERD", 1);

  LOG("Initializing Qthreads runtime");
  if (qthread_initialize())
    FATAL("Could not initialize Qthreads runtime");
  LOG("Initialized Qthreads runtime");

  LOG("Number of CPUs = %d", getNumCPUs());
  LOG("Number of shepherds = %d", qthread_num_shepherds());
  LOG("Number of workers = %d", qthread_num_workers());
  LOG("Initialized Kitsune runtime (qthreads)");
}

/// Finalize kitsune's qthreads runtime, as well as the underlying Qthreads
/// runtime. This is intended to be called from a global destructor that is
/// generated by Kitsune. This is not thread-safe, but it is safe to call more
/// than once (subsequent calls will return immediately).
extern "C" void __kitqthr_finalize(void) {
  if (!__kitqthr_initialized()) {
    LOG("Cannot finalize runtime. Not initialized");
    return;
  }

  LOG("Finalizing Kitsune runtime (qthreads)");

  LOG("Finalizing Qthreads runtime");
  qthread_finalize();
  LOG("Finalized Qthreads runtime");

#ifdef KITRT_PAPI_ENABLED
  __kitpapi_finalize();
#endif // KITRT_PAPI_ENABLED

  KitQthrContext::delSingleton();

  LOG("Finalized Kitsune runtime (qthreads)");

  __kitrt_finalize();
}
