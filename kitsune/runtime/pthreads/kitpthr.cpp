//===- kitpthr.cpp - Kitsune runtime targeting C11 threads ----------------===//
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

#include "kitrt.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits.h>
#include <pthread.h>
#include <thread>

/// TODO: Move this to a different file because this could be used as a common
/// utility function to display varying levels of status messages from other
/// runtimes as well.
static void log(const char *label, const char *category, const char *msg,
                va_list args) {
  // TODO: It would be nice if we could colorize the label.
  if (label)
    fprintf(stderr, "%s: ", label);
  if (category)
    fprintf(stderr, "%s: ", category);
  vfprintf(stderr, msg, args);
  fprintf(stderr, "\n");
}

/// Print an error message to stderr and terminate the process with an exit
/// code. \param msg may be a printf-compatible format string. In that case,
/// any optional arguments must be of the appropriate types.
[[noreturn]] static void fatal(const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  log("kitpthr", "ERROR", msg, args);
  va_end(args);

  std::exit(EXIT_FAILURE);
}

/// Print a warning message to stderr. \param msg may be a printf-compatible
/// format string. In that case, any optional arguments must be of the
/// appropriate types.
static void warn(const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  log("kitpthr", "WARNING", msg, args);
  va_end(args);
}

/// Display an informational message if verbose mode has been enabled.
/// \param msg may be a printf-compatible format string. In this case, any
/// optional arguments must be of the appropriate type.
static void message(const char *msg, ...) {
  if (__kitrt_verbose_mode()) {
    va_list args;
    va_start(args, msg);
    log("kitpthr", nullptr, msg, args);
    va_end(args);
  }
}

[[noreturn]] static void kitpthrHandleCreateError(int err) {
  const char *lede = "Could not create thread";
  switch (err) {
  case EINVAL:
    fatal("%s. Invalid attributes", lede);
  case EAGAIN:
    fatal("%s. Insufficient resources", lede);
  case EPERM:
    fatal("%s. Insufficient permissions", lede);
  default:
    fatal("%s. Unknown error", lede);
  }
}

[[noreturn]] static void kitpthrHandleJoinError(int err) {
  const char *lede = "Error joining thread";
  switch (err) {
  case EDEADLK:
    fatal("%s. Deadlock detected", lede);
  case EINVAL:
    fatal("%s. Thread is not joinable", lede);
  case ESRCH:
    fatal("%s. Invalid thread id", lede);
  default:
    fatal("%s. Unknown error", lede);
  }
}

/// The type of the pthread start function.
using pthread_start_t = void *(*)(void *);

/// The function that is run on each thread. The last argument is a pointer to
/// a struct generated by the compiler where all the data needed by the
/// function is bundled together. start and end are the starts and ends of the
/// iteration space that must be split across the threads.
using KitPthrThrdFn = void (*)(int64_t start, int64_t end, int64_t gs,
                               void *args);

/// Metadata for each thread. This is passed to the thread launch function and
/// also contains the arguments to be passed to the "actual" thread function.
struct KitPthrThrdInfo {
  KitPthrThrdFn f;
  int64_t start;
  int64_t end;
  int64_t grainSize;
  void *args;
  pthread_t id;
  pthread_attr_t attr;
};

/// The thread context object. This is created and returned by
/// \ref __kitpthr_launch. This should be passed to __kitpthr_join where it
/// will be deleted.
struct KitPthrContext {
  /// The thread id's that were launched when this context was created.
  KitPthrThrdInfo *thrdInfo = nullptr;

  /// The number of threads that were launched.
  int64_t thrds = 0;

public:
  KitPthrContext() = default;
  KitPthrContext(int64_t thrds) {
    message("New context: %ld threads", thrds);
    this->thrdInfo = new KitPthrThrdInfo[thrds];
    this->thrds = thrds;
  }

  ~KitPthrContext() {
    message("Delete context");
    delete[] thrdInfo;
  }

  KitPthrContext(const KitPthrContext &) = delete;
  KitPthrContext(KitPthrContext &&) = delete;
  KitPthrContext &operator=(const KitPthrContext &) = delete;

public:
  /// The maximum number of threads supported by this runtime.
  static constexpr int64_t maxThreads = std::numeric_limits<uint32_t>::max();
};

/// Determine the number of threads to launch. If an environment variable named
/// KIT_NUM_THREADS is set to a positive decimal (base 10) integer, that value
/// will be returned. If the value of the environment variable is negative, or
/// if it cannot be interpreted as an integer, a warning is issued and the
/// number of processors available on the system is returned instead. If the
/// number of processors could not be determined, a warning message is displayed
/// and 1 is returned.
static int64_t kitpthrGetNumThreads(int64_t start, int64_t end) {
  if (const char *envNumThreads = getenv("KIT_NUM_THREADS")) {
    message("Environment variable KIT_NUM_THREADS=%s", envNumThreads);
    char *end = nullptr;
    long numThreads = strtol(envNumThreads, &end, 10);

    // If *end is not '\0', either there are additional non-numeric characters
    // in the environment variable, or the environment variable cannot be parsed
    // to a signed integer. In this case, ignore the environment variable.
    // If the value of the environment variable is a valid integer, it may
    // yet be out of range, in which case errno will be set to ERANGE.
    bool error = *end != '\0' || errno == ERANGE || numThreads < 0 ||
                 numThreads > KitPthrContext::maxThreads;
    if (!error)
      return numThreads;

    warn("Invalid number of threads in KIT_NUM_THREADS");
  }

  long numThreads = std::thread::hardware_concurrency();
  if (numThreads <= 0) {
    warn("Disable threading. Could not determine number of CPUs");
    return 1;
  } else if (numThreads > KitPthrContext::maxThreads) {
    warn("Disable threading: Too many CPUs found: %ld", numThreads);
    return 1;
  }

  message("CPUs on system: %ld", numThreads);
  return numThreads;
}

/// Run \param f on the main thread. Block until f completes. Always return
/// nullptr. \param start, \param end and \param args are passed to \param f.
static KitPthrContext *runOnMainThread(KitPthrThrdFn f, int64_t start,
                                       int64_t end, int64_t grainSize,
                                       void *args) {
  warn("Running on main thread");

  f(start, end, grainSize, args);

  return new KitPthrContext;
}

/// The function that is launched by each thread. This simply finds the "actual"
/// function that is to be run in \param thrdInfo and calls it. The arguments
/// arguments to the actual function are also present in \param thrdInfo. Always
/// returns 0.
static void *kitpthrThrdStartFn(KitPthrThrdInfo *thrdInfo) {
  KitPthrThrdFn f = thrdInfo->f;
  int64_t start = thrdInfo->start;
  int64_t end = thrdInfo->end;
  int64_t grainSize = thrdInfo->grainSize;
  void *args = thrdInfo->args;

  f(start, end, grainSize, args);

  return nullptr;
}

/// Launch some number of threads each of which will execute some number of
/// iterations in the space [\ref start, \ref end). Terminates the program with
/// a fatal error if any thread could not be launched. The number of threads to
/// launch is determined by the value of the KIT_NUM_THREADS environment
/// variable, if it is present and set to a valid positive integer, or the
/// number of CPU's on the system if either the environment variable is not set,
/// or if it is set to a positive integer. If the number of CPU's could not be
/// determined, or if only a single CPU is available, no threads are launched.
/// Instead, the \param f is run on the main thread, which will block until
/// \param f completes.
///
/// \param f The function to execute on each thread
/// \param start The start index of the iteration space
/// \param end The value one greater than the last index of the iteration space
/// \param grainSize The grainSize
/// \param args A struct containing data to be passed to \param f
/// \return An opaque thread context object. It is the caller's responsibility
/// to call \ref __kitpthr_join with this context object. If no threads are
/// launched, i.e. \param f is run on the main thread, nullptr will be returned
/// instead. In this case, the caller is not required to call
/// \ref __kitpthr_join.
extern "C" KitPthrContext *__kitpthr_launch(KitPthrThrdFn f, int64_t start,
                                            int64_t end, int64_t grainSize,
                                            void *args) {
  message("Launching");
  message("Iteration range: [%ld, %ld)", start, end);

  // This is the number of threads that *may* be launched. However, there may
  // not be enough for work for all threads, so the actual number of threads
  // that are launched may be smaller than this.
  int64_t availThrds = kitpthrGetNumThreads(start, end);

  // If only a single thread is to be used, don't launch any threads. Run f on
  // the main thread and block until it finishes. Return nullptr as the thread
  // context.
  if (availThrds == 1)
    return runOnMainThread(f, start, end, grainSize, args);

  int64_t thrdSpan = std::max((end - start) / availThrds, 1L);
  message("Iterations per thread: %ld", thrdSpan);

  // The actual number of threads that are to be launched. Adding `thrdSpan - 1`
  // to `end` nicely deals with the case where the range of iterations is not an
  // integer multiple of thrdSpan. We could have used a conditional statement to
  // handle the two cases separately, but this is a fairly standard way of doing
  // such things.
  int64_t thrds = (end + thrdSpan - 1 - start) / thrdSpan;
  message("Launching %ld threads", thrds);

  KitPthrContext *ctx = new KitPthrContext(thrds);
  for (int64_t i = 0, beg = start; beg < end; beg += thrdSpan, ++i) {
    KitPthrThrdInfo &info = ctx->thrdInfo[i];
    info.f = f;
    info.start = beg;
    info.end = std::min(beg + thrdSpan, end);
    info.grainSize = grainSize;
    info.args = args;

    if (pthread_attr_init(&info.attr))
      fatal("Error initializing thread attrs");

    if (int err = pthread_create(&info.id, &info.attr,
                                 (pthread_start_t)kitpthrThrdStartFn,
                                 &ctx->thrdInfo[i]))
      kitpthrHandleCreateError(err);
    message("Launched [%ld, %ld) on thread %ld", info.start, info.end, i);
  }

  return ctx;
}

/// Join the threads launched by a previous call to \ref __kitpthr_launch.
/// \param ctx is the context returned by that call. \param ctx may be nullptr,
/// in which case, this function does nothing.
extern "C" void __kitpthr_sync(KitPthrContext *ctx) {
  message("Joining: %ld threads", ctx->thrds);
  for (size_t i = 0; i < ctx->thrds; ++i) {
    if (int err = pthread_join(ctx->thrdInfo[i].id, nullptr))
      kitpthrHandleJoinError(err);
    if (int err = pthread_attr_destroy(&ctx->thrdInfo[i].attr))
      fatal("Error destroying attributes");
    message("Joined thread %ld", i);
  }
  delete ctx;
}

/// Initialize kitsune's pthreads runtime. Currently, this only sets some
/// global variables that enable verbose mode. This runtime is not intended to
/// ever maintain any other state.
extern "C" void __kitpthr_initialize(void) {
  // Initialize the components of kitsune's runtime that are shared by the
  // tapir-target-specific components.
  __kitrt_initialize();
  message("Initialized runtime");
}

/// Finalize kitsune's pthreads runtime. This is only present for symmetry with
/// \ref __kitpthr_initialize. Since the runtime does not maintain any global
/// state of its own, this does nothing.
extern "C" void __kitpthr_finalize(void) { message("Finalize runtime"); }
