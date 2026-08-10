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

#include "pthreads/kitpthr.h"
#include "common/env.h"
#include "common/logging.h"
#include "common/utils.h"
#include "global/global.h"
#include "pthreads/context.h"

#include <cassert>
#include <cstring>
#include <errno.h>
#include <memory>
#include <pthread.h>
#include <vector>

using namespace kitrt;

namespace kitrt {

/// The type of the pthread start function.
using PthrStartFunc = void *(void *);

/// Metadata for each thread. This is passed to the thread launch function and
/// also contains the arguments to be passed to the "actual" thread function.
struct Pthread {
  PthrThrdFunc *f;
  int64_t tid;
  void *args;
  pthread_t pthr;
  pthread_attr_t attr;
};

/// The launch context object. An instance is created by
/// \ref __kitpthr_async_launch, This should be passed to __kitpthr_sync where
/// it will be deleted.
struct PthrLaunchContext {
  std::vector<Pthread> thrds;

  /// The argument bundle required by the functions that run on each thread.
  /// This is a copy of the bundle passed to __kitpthr_async_launch. It will be
  /// deleted when this is deleted, which will be when it is passed to
  /// __kitpthr_sync.
  std::unique_ptr<std::byte[]> thrdArgs;

  /// The size, in bytes, of the argument bundle.
  uint32_t thrdArgSize = 0;

public:
  PthrLaunchContext(size_t numThreads, void *args, uint32_t argSize)
      : thrds(numThreads) {
    for (size_t t = 0; t < numThreads; ++t)
      thrds[t].tid = t;
    if (args) {
      thrdArgs.reset(new std::byte[argSize]);
      memcpy(thrdArgs.get(), args, argSize);
      thrdArgSize = argSize;
    }
  }

  PthrLaunchContext(const PthrLaunchContext &) = delete;
  PthrLaunchContext(PthrLaunchContext &&) = delete;
  PthrLaunchContext &operator=(const PthrLaunchContext &) = delete;

  Pthread &operator[](size_t i) { return thrds.at(i); }
  const Pthread &operator[](size_t i) const { return thrds.at(i); }
  size_t size() const { return thrds.size(); }
  void *args() const { return thrdArgs.get(); }
  uint32_t argSize() const { return thrdArgSize; }

  decltype(thrds)::iterator begin() { return thrds.begin(); }
  decltype(thrds)::const_iterator begin() const { return thrds.begin(); }
  decltype(thrds)::iterator end() { return thrds.end(); }
  decltype(thrds)::const_iterator end() const { return thrds.end(); }
};

} // namespace kitrt

[[noreturn]] static void handleCreateError(int err) {
  const char *lede = "Could not create thread";
  switch (err) {
  case EINVAL: FATAL("%s. Invalid attributes", lede);
  case EAGAIN: FATAL("%s. Insufficient resources", lede);
  case EPERM: FATAL("%s. Insufficient permissions", lede);
  default: FATAL("%s. Unknown error", lede);
  }
}

[[noreturn]] static void handleJoinError(int err) {
  const char *lede = "Error joining thread";
  switch (err) {
  case EDEADLK: FATAL("%s. Deadlock detected", lede);
  case EINVAL: FATAL("%s. Thread is not joinable", lede);
  case ESRCH: FATAL("%s. Invalid thread id", lede);
  default: FATAL("%s. Unknown error", lede);
  }
}

void PthreadsContext::initialize() {
  numThreads = getNumThreadsOrCPUs();

  // pthreads does not have to be initialized.

  LOG("Number of CPUs = %d", getNumCPUs());
  LOG("Number of threads = %d", numThreads);
}

void PthreadsContext::finalize() {
  // pthreads does not need to be finalized.
}

uint64_t PthreadsContext::getNumThreads() const { return numThreads; }

KitThreadID PthreadsContext::getThreadID() const { return pthread_self(); }

/// The function that is launched by each thread. This simply finds the "actual"
/// function that is to be run in \p thrdInfo and calls it. The arguments to the
/// actual function are also present in \p thrdInfo. Always returns 0.
static void *launchOnThread(Pthread *thread) {
  KitPthrThrdFunc f = thread->f;
  int64_t tid = thread->tid;
  void *args = thread->args;

  f(tid, tid + 1, args);

  return nullptr;
}

PthrLaunchContext *PthreadsContext::launch(PthrThrdFunc *f, uint64_t start,
                                           uint64_t end, void *args,
                                           uint32_t argSize) {
  assert(start == 0 && end == numThreads &&
         "__kitpthr_async_launch expects loop iterations in range [0, N)");
  LOG("Launching multithreaded loop: [%ld,%ld)", start, end);

  PthrLaunchContext *ctx = new PthrLaunchContext(numThreads - 1, args, argSize);
  for (Pthread &thrd : *ctx) {
    thrd.f = f;
    thrd.args = ctx->args();

    if (pthread_attr_init(&thrd.attr))
      FATAL("Error initializing thread attributes");

    if (int err = pthread_create(&thrd.pthr, &thrd.attr,
                                 (PthrStartFunc *)launchOnThread, &thrd))
      handleCreateError(err);
    LOG("Fork thread %ld (%ld)", thrd.tid, thrd.pthr);
  }
  f(numThreads - 1, numThreads, ctx->args());

  return ctx;
}

void PthreadsContext::sync(PthrLaunchContext *ctx) {
  LOG("Joining %ld threads", ctx->size());
  for (Pthread &thrd : *ctx) {
    if (int err = pthread_join(thrd.pthr, nullptr))
      handleJoinError(err);
    if (pthread_attr_destroy(&thrd.attr))
      FATAL("Error destroying thread attributes");
    LOG("Joined thread %ld (%ld)", thrd.tid, thrd.pthr);
  }
  delete ctx;
}

// -----------------------------------------------------------------------------
// Everything below this is the public interface.

extern "C" uint64_t __kitpthr_num_threads(void) {
  return getCtx<PthreadsContext>().getNumThreads();
}

extern "C" KitThreadID __kitpthr_thread_id(void) {
  return getCtx<PthreadsContext>().getThreadID();
}

extern "C" KitPthrLaunchContext *
__kitpthr_async_launch(KitPthrThrdFunc f, uint64_t start, uint64_t end,
                       void *args, uint32_t argSize) {
  PthrLaunchContext *handle =
      mutCtx<PthreadsContext>().launch(f, start, end, args, argSize);
  return reinterpret_cast<KitPthrLaunchContext *>(handle);
}

extern "C" void __kitpthr_sync(KitPthrLaunchContext *handle) {
  if (auto *ctx = reinterpret_cast<PthrLaunchContext *>(handle))
    mutCtx<PthreadsContext>().sync(ctx);
}
