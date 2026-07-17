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

#include "kitocilk.h"
#include "common/env.h"
#include "common/logging.h"
#include "kitrt.h"

#include <cassert>

using namespace kitrt;

/// Declare functions and globals from the opencilk runtime that are used here.
extern unsigned __cilkrts_nproc;
extern "C" unsigned __cilkrts_get_worker_number(void);
extern "C" unsigned __cilkrts_get_nworkers(void);
extern "C" void __cilkrts_internal_set_nworkers(unsigned nworkers);

namespace {

class KitOCilkSingleton;

static void newSingleton(void);
static void delSingleton(void);
static KitOCilkSingleton *getSingleton(void);

/// Global state for this runtime. We intentionally keep the members public
/// because it is not clear what advantage there is to hiding them.
class KitOCilkSingleton {
public:
  // Currently, there are no members. This runtime only needs to know if it has
  // been initialized. If the global singleton is not nullptr, then we know that
  // the runtime has been initialized.

private:
  KitOCilkSingleton() = default;
  ~KitOCilkSingleton() = default;

public:
  friend void newSingleton(void);
  friend void delSingleton(void);
};

/// FIXME: This should eventually be folded into a single global state object
/// for kitrt - whenever that happens. This will be created by
/// __kitocilk_initialize and deleted by __kitocilk_finalize. This object should
/// never be accessed directly. Instead, the *Singleton() functions should be
/// used.
static KitOCilkSingleton *gSingleton = nullptr;

static void newSingleton(void) { gSingleton = new KitOCilkSingleton(); }

static void delSingleton(void) {
  delete gSingleton;
  gSingleton = nullptr;
}

static KitOCilkSingleton *getSingleton(void) { return gSingleton; }

} // namespace

/// Get the number of workers available for parallel work. For consistency, this
/// function should be used when this must be queried instead of calling
/// `__cilkrts_get_nworkers` directly.
extern "C" uint64_t __kitocilk_num_workers(void) {
  assert(__kitocilk_initialized() && "kitocilk initialized");
  return __cilkrts_get_nworkers();
}

/// Get the ID of the worker from which this is called.
extern "C" uint64_t __kitocilk_worker_id(void) {
  return __cilkrts_get_worker_number();
}

/// The number of partial reductions to perform in parallel.
///
/// \param n The trip count of the parallel loop containing a reduction
extern "C" uint64_t __kitocilk_reduce_num_partials(uint64_t n) {
  assert(__kitocilk_initialized() && "kitocilk initialized");
  LOG("Calculating number of partial reductions");

  // There might be something smarter that can be done once we support a proper
  // reduction tree, but since we only support a reduction tree of depth 1, we
  // just return the number of available workers.
  uint64_t numPartials = __kitocilk_num_workers();

  LOG("Number of partial reductions: %d", numPartials);

  return numPartials;
}

/// Check if this runtime has already been initialized.
extern "C" bool __kitocilk_initialized(void) { return getSingleton(); }

/// Initialize kitsune's OpenCilk runtime.
extern "C" void __kitocilk_initialize(void) {
  if (__kitocilk_initialized()) {
    LOG("Runtime already initialized");
    return;
  }

  LOGEARLY("Initializing Kitsune runtime (opencilk)");

  // Create the global singleton object.
  newSingleton();

  // Initialize the components of kitsune's runtime that are shared by the
  // tapir-target-specific components.
  __kitrt_initialize();

#ifdef KITRT_PAPI_ENABLED
  __kitpapi_initialize(__kitocilk_worker_id);
#endif // KITRT_PAPI_ENABLED

  uint64_t numThreads = __kitrt_num_threads("CILK_NWORKERS");

  // If the OpenCilk runtime has already been initialized, the number of workers
  // will have been set to the number of CPU's detected on the system. In that
  // case, there is no way to increase the number of workers, though it is still
  // possible to limit the number of workers to a value less than the number of
  // CPU's. Since we cannot control the order in which the initializers are run,
  // for consistency, we always disallow increasing the number of workers beyond
  // the number of CPU's.
  uint64_t numCPUs = __kitrt_num_cpus();
  if (numThreads > __kitrt_num_cpus())
    FATAL("Number of threads/workers (%d) cannot be greater than number of "
          "detected CPUs (%d)",
          numThreads, numCPUs);

  // Both of the lines below are required. __cilkrts_nproc is returned by
  // __cilkrts_nworkers. But it is not set by __cilkrts_internal_set_nworkers.
  // If we only call set_nworkers, parallel for loops will use the correct
  // number of threads, but reductions will not. If we only set __cilkrts_nproc,
  // reductions will perform the correct number of parallel reductions, but
  // parallel for loops will not use the correct number of threads.
  __cilkrts_nproc = numThreads;
  __cilkrts_internal_set_nworkers(numThreads);

  // There is no way to initialize OpenCilk's runtime explicitly - it is
  // initialized by its own private global constructor. We have no control over
  // when that constructor is run relative to this one.

  LOG("Number of workers = %d", __kitocilk_num_workers());
  LOG("Initialized Kitsune runtime (opencilk)");
}

/// Finalize kitsune's OpenCilk runtime.
extern "C" void __kitocilk_finalize(void) {
  if (!__kitocilk_initialized()) {
    LOG("Cannot finalize runtime. Not initialized");
    return;
  }

  LOG("Finalizing Kitsune runtime (opencilk)");

  // There is no way to finalize OpenCilk's runtime explicitly - it is finalized
  // by its own private global destructor. We have no control over when that
  // destructor is run relative to this one.

  // Finalize the components of Kitsune's runtime that are shared by the
  // tapir-target-specific components.
  __kitrt_finalize();

  // Delete the global singleton object.
  delSingleton();

  LOG("Finalized Kitsune runtime (opencilk)");
}
