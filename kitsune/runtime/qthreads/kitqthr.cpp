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

#include "qthreads/kitqthr.h"
#include "common/env.h"
#include "common/logging.h"
#include "common/utils.h"
#include "global/global.h"
#include "qthreads/context.h"

#include <qthread.h>
#include <qthread/barrier.h>

#include <cassert>
#include <vector>

using namespace kitrt;

namespace {

/// The arguments is passed to the thread launch function. This contains the
/// arguments to be passed to the "actual" thread function. It also contains the
/// barrier that must be entered once the actual thread function has finished
/// executing.
struct QthreadArgs {
  kitrt::QthrThrdFunc *f;
  int64_t tid;
  void *args;
  qt_barrier_t *barrier;
};

} // namespace

void QthreadsContext::initialize() {
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
}

void QthreadsContext::finalize() {
  LOG("Finalizing Qthreads runtime");
  qthread_finalize();
  LOG("Finalized Qthreads runtime");
}

uint64_t QthreadsContext::getNumThreads() const {
  return qthread_num_workers();
}

KitThreadID QthreadsContext::getThreadID() const { return qthread_id(); }

/// The function that is launched by each thread. This simply finds the "actual"
/// function that is to be run in \p thrdArgs and calls it. The arguments to the
/// actual function are also present in \p thrdArgs. Always returns 0.
static unsigned long launchOnThread(QthreadArgs *thrdArgs) {
  KitQthrThrdFunc f = thrdArgs->f;
  int64_t tid = thrdArgs->tid;
  void *args = thrdArgs->args;
  qt_barrier_t *barrier = thrdArgs->barrier;

  f(tid, tid + 1, args);

  LOG("Thread entering barrier: %d", tid);
  qt_barrier_enter(barrier);

  return 0;
}

void QthreadsContext::launch(QthrThrdFunc *f, uint64_t start, uint64_t end,
                             void *args, [[maybe_unused]] uint32_t argSize) {
  assert(start == 0 && end == getNumThreads() &&
         "__kitqthr_launch expects loop iterations in range [0,NUM_THREADS)");
  LOG("Launching multithreaded loop: [%ld,%ld)", start, end);

  uint64_t numThrds = getNumThreads();

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

  std::vector<QthreadArgs> thrds(numThrds);
  for (uint64_t t = 0; t < numThrds; ++t) {
    QthreadArgs &thrdArgs = thrds[t];
    thrdArgs.f = f;
    thrdArgs.tid = t;
    thrdArgs.args = args;
    thrdArgs.barrier = barrier;

    LOG("Fork thread %d", t);
    if (qthread_fork((qthread_f)launchOnThread, &thrdArgs, nullptr))
      FATAL("Could not fork thread");
  }

  // The main thread must also enter the barrier. Once the main thread has
  // entered, it will block until all spawned threads enter before continuing.
  qt_barrier_enter(barrier);
  qt_barrier_destroy(barrier);

  LOG("Finished multithreaded loop");
}

// -----------------------------------------------------------------------------
// Everything below this is the public interface.

extern "C" uint64_t __kitqthr_num_workers(void) {
  return getCtx<QthreadsContext>().getNumThreads();
}

extern "C" KitThreadID __kitqthr_worker_id(void) {
  return getCtx<QthreadsContext>().getThreadID();
}

extern "C" void __kitqthr_launch(KitQthrThrdFunc f, uint64_t start,
                                 uint64_t end, void *args,
                                 [[maybe_unused]] uint32_t argSize) {
  return mutCtx<QthreadsContext>().launch(f, start, end, args, argSize);
}
