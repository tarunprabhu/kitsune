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
#include "global/global.h"

#include <cstdlib>

using namespace kitrt;

extern "C" bool __kitrt_initialized(void) { return gctx.initialized; }

extern "C" void __kitrt_initialize(void) {
  if (__kitrt_initialized()) {
    LOG("Runtime already initialized");
    return;
  }

  // At the point, the global singleton object has not yet been initialized.
  // AS a result, verbose mode will always return false. The only way to ensure
  // that a message is printed is to check the environment variables.
  LOG_IF_VERBOSE("Initializing Kitsune runtime (common)");

  KitRTContext &rt = mutKitRTContext();
  rt.setVerbose(envLookupOr(envVerbose, envVerboseLegacy, false));

  __kittimer_initialize();

  rt.setInitialized(true);
  LOG("Initialized Kitsune runtime (common)");
}

extern "C" void __kitrt_finalize(void) {
  if (!__kitrt_initialized()) {
    LOG("Cannot finalize runtime. Not initialized");
    return;
  }

  LOG("Finalizing Kitsune runtime (common)");

  __kittimer_finalize();

  mutKitRTContext().setInitialized(false);

  // Although we don't do so, it is reasonable to clear the global singleton at
  // this point. In that case, LOG(...) would not work because verbose mode
  // would always be false. Using LOG_IF_VERBOSE(...) here serves as a marker
  // that the global context object should not be used beyond this point.
  LOG_IF_VERBOSE("Finalized Kitsune runtime (common)");
}
