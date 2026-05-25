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

#include "kitrt.h"

// This is an internal header in LLVM's OpenMP runtime. The path is relative
// to ${LLVM_MONOREPO_SOURCE_DIR}/openmp/runtime/src.
#include "kmp.h"

#include <cstdint>
#include <string_view>

#define LABEL "kitomp"

// Not all functions are exposed in the internal header kmp.h. These are used by
// clang when generating LLVM-IR, so it is "reasonable" to use these here.
// For consistency, we may want to forward declare everything that we need from
// that header and just dropping the include.
extern "C" void __kmpc_for_static_init_8(ident_t *loc, int32_t gtid,
                                         int32_t schedtype, int32_t *plastiter,
                                         int64_t *plower, int64_t *pupper,
                                         int64_t *pstride, int64_t incr,
                                         int64_t chunk);

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

/// The number of threads to use. Ideally, this should be part of a global
/// object that contains all the state needed by the runtime. But that would
/// require reorganization of the runtime. A separate effort is underway that
/// does this, so do this temporarily for now.
///
/// A better approach would be to use the libomp API, but the functions there
/// only seem to work as expected from within a parallel region. However, we
/// often need to get the number of threads from a serial context.
///
/// This will be set in the global ctor for this runtime. This should not be
/// used directly in the rest of the runtime. Use `__kitomp_num_threads()` to
/// get this value.
static unsigned __kitomp_num_threads_v = 1;

/// Get the number of parallel workers that are available. Generally, this
/// function should be used when this must be queried instead of calling
/// `qthread_num_workers()`.
static unsigned __kitomp_num_threads() { return __kitomp_num_threads_v; }

/// The function that is run on each thread. The last argument is a pointer to
/// a struct generated by the compiler where all the data needed by the
/// function is bundled together. start and end are the starts and ends of the
/// iteration space that must be split across the threads.
using KitOpenMPThrdFn = void (*)(const int64_t start, const int64_t end,
                                 const int64_t gs, void *args);

/// This wraps the function \p f that will be launched on each thread. It
/// calculates the range of iterations that should be executed by \p f, then,
/// invokes it with that range. This uses static scheduling - so every thread
/// will execute a roughly equal subset of the iterations.
///
/// The exact subset of the iterations to calculate on a thread is determined
/// by libomp functions that are called here.
static void staticLoopWrapper(int32_t *globalTID, int32_t *localTID,
                              KitOpenMPThrdFn f, int64_t start, int64_t end,
                              int64_t grainSize, void *args) {
  // These functions take an inclusive range of iterations. Consider the loop
  // below:
  //
  //     forall (long i = 0; i < n; ++i)
  //         ...
  //
  // Here, the loop bounds are [0, n) since i will never be equal to `n - 1` in
  // the loop body. Since the libomp functions expect - and return - inclusive
  // ranges. the upper bound is set to `n - 1`.
  //
  int32_t lastIter = 0;
  int64_t lower = start;
  int64_t upper = end - 1;
  int64_t stride = 1;

  // Calculate the subset of iterations to be executed by this thread. This will
  // return an inclusive range in lower and upper i.e. upper will be the last
  // iteration to be executed on this thread - not the value one past the last
  // iteration.
  __kmpc_for_static_init_8(&staticLoopLoc, *globalTID, kmp_sch_static,
                           &lastIter, &lower, &upper, &stride, /*incr=*/1,
                           /*chunk=*/1);

  // Since upper will be the last iteration to be executed on the thread, and
  // the body of f is guaranteed to be of the form `i < UPPER_BOUND`, we pass
  // upper + 1 to f.
  __kitrt_message(LABEL, "Running on thread %d:%d: [%ld, %ld)", *globalTID,
                  *localTID, lower, upper + 1);
  if (lower <= upper)
    f(lower, upper + 1, grainSize, args);

  __kmpc_for_static_fini(&staticLoopLoc, *globalTID);
}

/// Launch some number of threads each of which will execute some number of
/// iterations in the space [\ref start, \ref end). This blocks until all
/// threads have completed.
///
/// \param f The function to execute on each thread
/// \param start The start index of the iteration space
/// \param end The value one greater than the last index of the iteration space
/// \param grainSize The grain size. This is not used and is expected to be 0.
/// \param args Pointer to the struct containing data to be passed to \p f
extern "C" void __kitomp_launch(KitOpenMPThrdFn f, int64_t start, int64_t end,
                                int64_t grainSize, void *args) {
  __kitrt_message(LABEL, "Launching multithreaded loop: [%ld, %ld)", start,
                  end);
  __kmpc_fork_call(&staticLoopLoc, 5, (kmpc_micro)&staticLoopWrapper, f, start,
                   end, grainSize, args);
  __kitrt_message(LABEL, "Finished multithreaded loop");
}

/// The number of partial reductions to perform in parallel.
///
/// \param n The trip count of the parallel loop in containing a reduction
extern "C" int64_t __kitomp_reduce_num_partials(int64_t n) {
  __kitrt_message(LABEL, "Calculating number of partial reductions");

  // There might be something smarter that can be done once we support a proper
  // reduction tree, but since we only support a reduction tree of depth 1, we
  // just return the number of CPU's on the system.
  unsigned numPartials = __kitomp_num_threads();

  __kitrt_message(LABEL, "Number of partial reductions: %d", numPartials);

  return numPartials;
}

/// Initialize kitsune's OpenMP runtime as well as the actual OpenMP runtime.
extern "C" void __kitomp_initialize(void) {
  // Initialize the components of kitsune's runtime that are shared by the
  // tapir-target-specific components.
  __kitrt_initialize();
  __kitrt_message(LABEL, "Initializing Kitsune openmp runtime");

  if (unsigned numThreads = __kitrt_num_threads_from_env()) {
    __kitomp_num_threads_v = numThreads;

    const char *s = getenv(__kitrt_envname_num_threads);
    __kitrt_message(LABEL, "Setting OMP_NUM_THREADS=%s", s);
    __kitrt_set_env("OMP_NUM_THREADS", s);
  } else {
    __kitomp_num_threads_v = __kitrt_num_cpus();
  }

  // The second argument in the call to __kmpc_begin is currently unused, per
  // the 10-year old documentation that seems to be the only kind that is
  // available.
  __kitrt_message(LABEL, "Initializing OpenMP runtime");
  __kmpc_begin(&unknownLoc, /*flags=*/0);
  __kitrt_message(LABEL, "Initialized OpenMP runtime");

  __kitrt_message(LABEL, "Number of threads = %d", __kitomp_num_threads());
  __kitrt_message(LABEL, "Initialized Kitsune openmp runtime");
}

/// Finalize kitsune's OpenMP runtime, as well as OpenMP runtime.
extern "C" void __kitomp_finalize(void) {
  __kitrt_message(LABEL, "Finalizing Kitsune openmp runtime");

  // This call is optional, but we use it anyway for consistency with the other
  // runtimes.
  __kitrt_message(LABEL, "Finalizing OpenMP runtime");
  __kmpc_end(&unknownLoc);
  __kitrt_message(LABEL, "Finalized OpenMP runtime");

  __kitrt_message(LABEL, "Finalized Kitsune openmp runtime");
}
