//===- utlis.cpp - Miscellaneous utilities --------------------------------===//
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
//
// Some useful utilities that are used by several tapir-target-specific
// runtimes.
//
// TODO: Rather than just dumping a number of unrelated utilities here, we
// should try to find a more reasonable organization across source files.
//
//===----------------------------------------------------------------------===//

#include "common/utils.h"
#include "common/env.h"
#include "common/logging.h"

#include <execinfo.h>
#include <thread>

using namespace kitrt;

uint32_t kitrt::getNumThreadsOrCPUs(const char *alternate) {
  auto warnIfExists = [](const char *envVar) -> void {
    if (envContains(envVar))
      WARN("Ignoring environment variable '%s' with invalid value \"%s\"",
           envVar, envLookup(envVar)->c_str());
  };

  if (std::optional<uint32_t> threads = envLookup<uint32_t>(envNumThreads)) {
    if (*threads > 0) {
      LOG("Environment contains %s=%d", envNumThreads, *threads);
      return *threads;
    }
  }

  // If the environment variable was set, but to an invalid value, issue a
  // diagnostic since the behavior can be confusing if kitrt silently ignores
  // variable and the user does not realize their mistake.
  warnIfExists(envNumThreads);

  if (alternate) {
    if (std::optional<uint32_t> threads = envLookup<uint32_t>(alternate)) {
      // If the value is set to zero, don't use it. This is just to be
      // consistent with the behavior of KIT_NUM_THREADS. It is possible that
      // the underlying runtime can handle the value, but we don't want to take
      // that chance.
      if (*threads > 0) {
        LOG("Environment contains %s=%d", alternate, *threads);
        return *threads;
      }
    }
    warnIfExists(alternate);
  }

  return getNumCPUs();
}

uint32_t kitrt::getNumCPUs(void) {
  // The standard says that std::thread::hardware_concurrency() should only be
  // considered a hint. But it seems to work on the platforms that we care
  // about. Still, it might be worth using a more reliable method.
  unsigned cpus = std::thread::hardware_concurrency();
  if (cpus == 0) {
    WARN("Could not determine number of CPUs. Defaulting to 1");
    return 1;
  }

  LOG("Found %d CPUs", cpus);
  return cpus;
}

void kitrt::printStackTrace(void) {
  constexpr int depth = 25;
  void *trace[depth];
  int size = backtrace(trace, depth);
  if (char **strings = backtrace_symbols(trace, size)) {
    LOG("stack trace (%d frames)", size);
    for (int i = 0; i < size; ++i)
      LOG("  %s", strings[i]);
    LOG("end stack trace");
    free(strings);
  }
}
