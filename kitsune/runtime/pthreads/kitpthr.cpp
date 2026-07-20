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
#include "common/utils.h"
#include "global/singleton.h"
#include "kitrt.h"

#include <cassert>
#include <cstring>
#include <errno.h>
#include <memory>
#include <pthread.h>
#include <vector>

using namespace kitrt;

namespace kitrt {

/// Global state for this runtime. We intentionally keep the members public
/// because it is not clear what advantage there is to hiding them.
class KitPthrContext : public KitContextMixin<KitPthrContext> {
public:
  /// The number of threads to use. This should not be used directly in the rest
  /// of the runtime. Use `__kitpthr_num_threads()` to get this value.
  uint32_t numThreads = 1;
};

} // namespace kitrt

namespace {

/// The type of the pthread start function.
using PthreadStartFunc = void *(*)(void *);

/// Metadata for each thread. This is passed to the thread launch function and
/// also contains the arguments to be passed to the "actual" thread function.
struct KitPthrThread {
  KitPthrThrdFunc f;
  int64_t tid;
  void *args;
  pthread_t pthr;
  pthread_attr_t attr;
};

/// The launch context object. The instance created by \ref __kitpthr_launch
/// should be passed to __kitpthr_join where it will be deleted.
struct LaunchContextImpl {
  std::vector<KitPthrThread> thrds;

  /// The argument bundle required by the functions that run on each thread.
  /// This is a copy of the bundle passed to __kitpthr_launch. It will be
  /// deleted when this is deleted, which will be when it is passed to
  /// __kitpthr_sync.
  std::unique_ptr<std::byte[]> thrdArgs;

  /// The size, in bytes, of the argument bundle.
  uint64_t thrdArgSize = 0;

public:
  LaunchContextImpl(size_t numThreads, void *args, uint64_t argSize)
      : thrds(numThreads) {
    for (size_t t = 0; t < numThreads; ++t)
      thrds[t].tid = t;
    if (args) {
      thrdArgs.reset(new std::byte[argSize]);
      memcpy(thrdArgs.get(), args, argSize);
      thrdArgSize = argSize;
    }
  }

  LaunchContextImpl(const LaunchContextImpl &) = delete;
  LaunchContextImpl(LaunchContextImpl &&) = delete;
  LaunchContextImpl &operator=(const LaunchContextImpl &) = delete;

  KitPthrThread &operator[](size_t i) { return thrds.at(i); }
  const KitPthrThread &operator[](size_t i) const { return thrds.at(i); }
  size_t size() const { return thrds.size(); }
  void *args() const { return thrdArgs.get(); }
  uint64_t argSize() const { return thrdArgSize; }

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
    FATAL("%s. Invalid attributes", lede);
  case EAGAIN:
    FATAL("%s. Insufficient resources", lede);
  case EPERM:
    FATAL("%s. Insufficient permissions", lede);
  default:
    FATAL("%s. Unknown error", lede);
  }
}

[[noreturn]] static void kitpthrHandleJoinError(int err) {
  const char *lede = "Error joining thread";
  switch (err) {
  case EDEADLK:
    FATAL("%s. Deadlock detected", lede);
  case EINVAL:
    FATAL("%s. Thread is not joinable", lede);
  case ESRCH:
    FATAL("%s. Invalid thread id", lede);
  default:
    FATAL("%s. Unknown error", lede);
  }
}

/// Get the number of threads available for parallel work.
extern "C" uint64_t __kitpthr_num_threads(void) {
  assert(__kitpthr_initialized() && "kitpthr initialized");
  return KitPthrContext::getSingleton().numThreads;
}

/// Get the ID of the thread from which this is called.
extern "C" uint64_t __kitpthr_thread_id(void) { return pthread_self(); }

/// The number of partial reductions to perform in parallel.
///
/// \param n The trip count of the parallel loop containing a reduction
extern "C" uint64_t __kitpthr_reduce_num_partials(uint64_t n) {
  assert(__kitpthr_initialized() && "kitpthr initialized");
  LOG("Calculating number of partial reductions");

  // There might be something smarter that can be done once we support a proper
  // reduction tree, but since we only support a reduction tree of depth 1, we
  // just use as many partials as there are CPU's on the system.
  uint64_t numPartials = __kitpthr_num_threads();

  LOG("Number of partial reductions: %d", numPartials);

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
///     unsigned numThreads = __kitpthr_num_threads();
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
/// \params argSize The size of the underlying struct pointed to by \p args
/// \return An opaque thread context object. It is the caller's responsibility
/// to call \ref __kitpthr_join with this context object. If no threads are
/// launched, i.e. \p f is run on the main thread, nullptr will be returned
/// instead. In this case, the caller is not required to call
/// \ref __kitpthr_join.
extern "C" KitPthrLaunchContext *__kitpthr_launch(KitPthrThrdFunc f,
                                                  uint64_t start, uint64_t end,
                                                  void *args,
                                                  uint64_t argSize) {
  assert(__kitpthr_initialized() && "kitpthr initialized");
  assert(start == 0 && end == __kitpthr_num_threads() &&
         "__kitpthr_launch expects loop iterations in range [0,NUM_THREADS)");
  LOG("Launching multithreaded loop: [%ld,%ld)", start, end);

  uint64_t numThreads = __kitpthr_num_threads();
  LaunchContextImpl *ctx = new LaunchContextImpl(numThreads - 1, args, argSize);
  for (KitPthrThread &thrd : *ctx) {
    thrd.f = f;
    thrd.args = ctx->args();

    if (pthread_attr_init(&thrd.attr))
      FATAL("Error initializing thread attributes");

    if (int err = pthread_create(&thrd.pthr, &thrd.attr,
                                 (PthreadStartFunc)kitpthrThrdStartFn, &thrd))
      kitpthrHandleCreateError(err);
    LOG("Fork thread %ld (%ld)", thrd.tid, thrd.pthr);
  }
  f(numThreads - 1, numThreads, ctx->args());

  return reinterpret_cast<KitPthrLaunchContext *>(ctx);
}

/// Join the threads launched by a previous call to \ref __kitpthr_launch.
/// \p ctx is the context returned by that call. \p ctx may be nullptr, in which
/// case, this function does nothing.
extern "C" void __kitpthr_sync(KitPthrLaunchContext *p) {
  LaunchContextImpl *ctx = reinterpret_cast<LaunchContextImpl *>(p);
  LOG("Joining %ld threads", ctx->size());
  for (KitPthrThread &thrd : *ctx) {
    if (int err = pthread_join(thrd.pthr, nullptr))
      kitpthrHandleJoinError(err);
    if (pthread_attr_destroy(&thrd.attr))
      FATAL("Error destroying thread attributes");
    LOG("Joined thread %ld (%ld)", thrd.tid, thrd.pthr);
  }
  delete ctx;
}

/// Check if this runtime has already been initialized.
extern "C" bool __kitpthr_initialized(void) {
  return KitPthrContext::hasSingleton();
}

/// Initialize Kitsune's pthreads runtime. This is intended to be called from
/// a global constructor that is generated by Kitsune. This is not thread-safe,
/// but it is safe to call more than once (subsequent calls will return
/// immediately).
extern "C" void __kitpthr_initialize(void) {
  if (__kitpthr_initialized()) {
    LOG("Runtime already initialized");
    return;
  }

  __kitrt_initialize();

  LOG("Initializing Kitsune runtime (pthreads)");

  KitPthrContext::addSingleton();
  KitPthrContext::mutSingleton().numThreads = getNumThreadsOrCPUs();

#ifdef KITRT_PAPI_ENABLED
  __kitpapi_initialize(__kitpthr_thread_id);
#endif // KITRT_PAPI_ENABLED

  // pthreads does not have to be initialized.

  LOG("Number of CPUs = %d", getNumCPUs());
  LOG("Number of threads = %d", __kitpthr_num_threads());
  LOG("Initialized Kitsune runtime (pthreads)");
}

/// Finalize Kitsune's pthreads runtime. This is intended to be called from
/// a global destructor that is generated by Kitsune. This is not thread-safe,
/// but it is safe to call more than once (subsequent calls will return
/// immediately).
extern "C" void __kitpthr_finalize(void) {
  if (!__kitpthr_initialized()) {
    LOG("Cannot finalize runtime. Not initialized");
    return;
  }

  LOG("Finalizing Kitsune runtime (pthreads)");

  // pthreads does not need to be finalized.

#ifdef KITRT_PAPI_ENABLED
  __kitpapi_finalize();
#endif // KITRT_PAPI_ENABLED

  KitPthrContext::delSingleton();

  LOG("Finalized Kitsune runtime (pthreads)");

  __kitrt_finalize();
}
