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

  __kittimer_finalize();

  __kitrt_finalized = true;
  LOG("Finalized Kitsune runtime (common)");
}
