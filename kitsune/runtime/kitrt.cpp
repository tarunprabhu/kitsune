//===- kitrt.cpp - Routines common to several of Kitsune's runtimes -------===//
//
// Copyright (c) 2021, Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021. Los Alamos National Security, LLC. This software was
//  produced under U.S. Government contract DE-AC52-06NA25396 for Los
//  Alamos National Laboratory (LANL), which is operated by Los Alamos
//  National Security, LLC for the U.S. Department of Energy. The
//  U.S. Government has rights to use, reproduce, and distribute this
//  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
//  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
//  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
//  derivative works, such modified software should be clearly marked,
//  so as not to confuse it with the version available from LANL.
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
#include "common/env.h"
#include "common/logging.h"

#include <cstdlib>
#include <execinfo.h>
#include <thread>

using namespace kitrt;

// FIXME: Combine all global variables here into a single struct.
static bool __kitrt_initialized = false;
static bool __kitrt_finalized = false;

// This should be private. However, we expose it because it is examined often by
// most runtimes to determine whether to print informational messages. Wrapping
// it in a function may be expensive.
bool _kitrt_verbose_mode = false;

extern "C" void __kitrt_initialize(void) {
  if (__kitrt_initialized)
    return;

  LOGEARLY("Initializing Kitsune runtime (common)");

  _kitrt_verbose_mode = envLookupOr(envVerbose, envVerboseLegacy, false);

  // This message will only be printed if verbose mode if _kitrt_verbose_mode
  // gets set to true above.
  LOG("Verbose mode enabled");

  __kittimer_initialize();

  __kitrt_initialized = true;
  LOG("Initialized Kitsune runtime (common)");
}

extern "C" void __kitrt_finalize(void) {
  if (__kitrt_finalized)
    return;

  LOG("Finalizing Kitsune runtime (common)");

#ifdef KITRT_PAPI_ENABLED
  __kitpapi_finalize();
#endif // KITRT_PAPI_ENABLED
  __kittimer_finalize();

  __kitrt_finalized = true;
  LOG("Finalized Kitsune runtime (common)");
}

extern "C" void __kitrt_print_stack_trace(void) {
  const unsigned depth = 25;
  void *trace[depth];
  int size = backtrace(trace, depth);
  if (char **strings = backtrace_symbols(trace, size)) {
    LOG("stack trace (%d frames)", size);
    for (int i = 0; i < size; i++)
      LOG("  %s", strings[i]);
    LOG("end stack trace");
    free(strings);
  }
}

uint32_t nearestPowerOf2LE(uint32_t n) {
  uint64_t p = 1;
  while (p <= n)
    p <<= 1;
  return p >> 1;
}

extern "C" uint32_t __kitrt_num_threads(const char *alternate) {
  if (std::optional<uint32_t> threads = envLookup<uint32_t>(envNumThreads)) {
    LOG("Environment contains %s=%d", envNumThreads, *threads);
    return *threads;
  }

  if (alternate) {
    if (std::optional<uint32_t> threads = envLookup<uint32_t>(alternate)) {
      LOG("Environment contains %s=%d", alternate, *threads);
      return *threads;
    }
  }

  return __kitrt_num_cpus();
}

extern "C" uint32_t __kitrt_num_cpus(void) {
  LOG("Determining number of CPUs");

  // The standard says that std::thread::hardware_concurrency() should only be
  // considered a hint. But it seems to work on the platforms that we care
  // about. Still, it might be worth using a more reliable method.
  unsigned cpus = std::thread::hardware_concurrency();
  if (cpus == 0) {
    WARN("Could not determine number of CPUs");
    return 1;
  }

  LOG("Found %d CPUs", cpus);
  return cpus;
}
