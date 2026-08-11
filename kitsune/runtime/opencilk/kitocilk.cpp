//===- kitocilk.cpp - Runtime for Kitsune's opencilk tapir target ---------===//
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
// This targets Cheetah, the OpenCilk runtime.
//
// The opencilk tapir target does nearly all the work of interfacing with
// Cheetah. Most of the functions here exist mainly to simplify the lowering of
// Kitsune's intrinsics.
//
//===----------------------------------------------------------------------===//

#include "opencilk/kitocilk.h"
#include "common/env.h"
#include "common/logging.h"
#include "common/utils.h"
#include "context.h"
#include "opencilk/context.h"

#include <cassert>

/// Declare functions and globals from the opencilk runtime that are used here.
extern unsigned __cilkrts_nproc;
extern "C" unsigned __cilkrts_get_worker_number(void);
extern "C" unsigned __cilkrts_get_nworkers(void);
extern "C" void __cilkrts_internal_set_nworkers(unsigned nworkers);
extern "C" int __cilkrts_is_initialized(void);

using namespace kitrt;

/// Initialize Kitsune's OpenCilk runtime. This does *not* initialize the
/// underlying OpenCilk runtime. That happens independently of this, and may
/// happen before or after this.
void OpenCilkContext::initialize() {
  uint64_t numThreads = getNumThreadsOrCPUs("CILK_NWORKERS");
  if (__cilkrts_is_initialized()) {
    LOG("OpenCilk runtime has already been initialized");
    LOG("Overriding number of workers");

    // If the OpenCilk runtime has already been initialized, the number of
    // workers will have been set to the number of CPU's detected on the system.
    // In that case, there is no way to increase the number of workers, though
    // it is still possible to limit the number of workers to a value less than
    // the number of CPU's. Since we cannot control the order in which the
    // initializers are run, for consistency, we always disallow increasing the
    // number of workers beyond the number of CPU's.
    uint64_t numCPUs = getNumCPUs();
    if (numThreads > numCPUs)
      FATAL("Number of threads/workers (%d) cannot be greater than number of "
            "detected CPUs (%d)",
            numThreads, numCPUs);

    // Both of the lines below are required. __cilkrts_nproc is returned by
    // __cilkrts_nworkers. But it is not set by __cilkrts_internal_set_nworkers.
    // If we only call set_nworkers, parallel for loops will use the correct
    // number of threads, but reductions will not. If we only set
    // __cilkrts_nproc, reductions will perform the correct number of parallel
    // reductions, but parallel for loops will not use the correct number of
    // threads.
    __cilkrts_nproc = numThreads;
    __cilkrts_internal_set_nworkers(numThreads);
  } else {
    // If the OpenCilk runtime has not already been initialized, the simply
    // setting CILK_NWORKERS in the environment should be sufficient. When the
    // global ctor for Cheetah runs, it will setup the number of workers
    // correctly.
    LOG("OpenCilk runtime has not been initialized");
    envSet("CILK_NWORKERS", numThreads);
  }

  // There is no way to initialize OpenCilk's runtime explicitly - it is
  // initialized by its own private global constructor. We have no control over
  // when that constructor is run relative to this one.

  LOG("Number of CPUs = %d", getNumCPUs());
  LOG("Number of workers = %d", getNumThreads());
}

void OpenCilkContext::finalize() {
  // There is no way to finalize OpenCilk's runtime explicitly - it is finalized
  // by its own private global destructor. We have no control over when that
  // destructor is run relative to this one.
}

uint64_t OpenCilkContext::getNumThreads() const {
  return __cilkrts_get_nworkers();
}

KitThreadID OpenCilkContext::getThreadID() const {
  return __cilkrts_get_worker_number();
}

// -----------------------------------------------------------------------------
// Everything below this is the public interface.

extern "C" uint64_t __kitocilk_num_workers(void) {
  return getCtx<OpenCilkContext>().getNumThreads();
}

extern "C" KitThreadID __kitocilk_worker_id(void) {
  return getCtx<OpenCilkContext>().getThreadID();
}
