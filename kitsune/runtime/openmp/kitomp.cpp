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
#include "common/utils.h"

// This is an internal header in LLVM's OpenMP runtime. The path is relative
// to ${LLVM_MONOREPO_SOURCE_DIR}/openmp/runtime/src.
#include "kmp.h"

#include <cassert>
#include <string_view>

// Since we have included kmp.h, the implementation header for OpenMP, we cannot
// also include omp.h. Therefore, we have to redeclare the functions from omp.h
// that we use.
extern "C" unsigned omp_get_thread_num(void);
extern "C" unsigned omp_get_num_threads(void);
extern "C" unsigned omp_get_max_threads(void);

using namespace kitrt;

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

void KitOMPContext::initialize() {
  uint64_t numThreads = getNumThreadsOrCPUs("OMP_NUM_THREADS");
  envSet("OMP_NUM_THREADS", numThreads);

  // The second argument in the call to __kmpc_begin is currently unused, per
  // the 10-year old documentation that seems to be the only kind that is
  // available.
  LOG("Initializing OpenMP runtime");
  __kmpc_begin(&unknownLoc, /*flags=*/0);
  LOG("Initialized OpenMP runtime");

  LOG("Number of CPUs = %d", getNumCPUs());
  LOG("Number of threads = %d", __kitomp_num_threads());
}

void KitOMPContext::finalize() {
  // This call is optional, but we use it anyway for consistency with the other
  // runtimes.
  LOG("Finalizing OpenMP runtime");
  __kmpc_end(&unknownLoc);
  LOG("Finalized OpenMP runtime");
}

uint64_t KitOMPContext::getNumThreads() const { return omp_get_max_threads(); }

KitThreadID KitOMPContext::getThreadID() const { return omp_get_thread_num(); }

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
  LOG("Running on thread %d of %d [global = %d]", *localTID,
      omp_get_num_threads(), *globalTID);

  // Since each thread must run exactly one iteration, we can just use the
  // thread id to specify the range.
  f(*localTID, *localTID + 1, args);
}

void KitOMPContext::launch(KitOMPThrdFunc *f, uint64_t start, uint64_t end,
                           void *args, [[maybe_unused]] uint32_t argSize) {
  assert(start == 0 && end == getNumThreads() &&
         "__kitomp_launch expects loop iterations in range [0,NUM_THREADS)");
  LOG("Launching multithreaded loop: [%ld,%ld)", start, end);

  // This will launch `omp_get_max_threads()` threads.
  __kmpc_fork_call(&staticLoopLoc, 4, (kmpc_micro)&staticLoopWrapper, f, start,
                   end, args);

  LOG("Finished multithreaded loop");
}

extern "C" uint64_t __kitomp_num_threads(void) {
  KitOMPContext::ensure();
  return KitOMPContext::get().getNumThreads();
}

extern "C" KitThreadID __kitomp_thread_id(void) {
  KitOMPContext::ensure();
  return KitOMPContext::get().getThreadID();
}

extern "C" void __kitomp_launch(KitOMPThrdFunc *f, uint64_t start, uint64_t end,
                                void *args, [[maybe_unused]] uint32_t argSize) {
  KitOMPContext::ensure();
  KitOMPContext::mut().launch(f, start, end, args, argSize);
}
