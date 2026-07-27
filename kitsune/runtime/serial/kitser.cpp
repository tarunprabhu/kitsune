//===- kitser.cpp - Runtime for Kitsune's serial tapir target -------------===//
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
// This does very, very little. It's main purpose is to ensure that the
// dependencies, such as PAPI, are initialized when the serial tapir target is
// used.
//
//===----------------------------------------------------------------------===//

#include "kitser.h"
#include "common/logging.h"
#include "global/singleton.h"
#include "kitrt.h"

using namespace kitrt;

namespace kitrt {

/// Global state for this runtime. We intentionally keep the members public
/// because it is not clear what advantage there is to hiding them.
class KitSerContext : public KitContextMixin<KitSerContext> {
public:
  // Currently, there are no members. This runtime only needs to know if it has
  // been initialized. If the global singleton is not nullptr, then we know that
  // the runtime has been initialized.
};

} // namespace kitrt

/// Get the ID of the thread from which this is called. Always return 0.
extern "C" KitThreadID __kitser_thread_id(void) { return 0; }

/// Check if this runtime has already been initialized.
extern "C" bool __kitser_initialized(void) {
  return KitSerContext::hasSingleton();
}

/// Initialize Kitsune's serial runtime. This is not expected to do much other
/// than initialize the rest of the runtime and initialize PAPI.
extern "C" void __kitser_initialize(void) {
  if (__kitser_initialized()) {
    LOG("Runtime already initialized");
    return;
  }

  __kitrt_initialize();

  LOG("Initializing Kitsune runtime (serial)");
  KitSerContext::addSingleton();
#ifdef KITRT_PAPI_ENABLED
  __kitpapi_initialize(__kitser_thread_id);
#endif // KITRT_PAPI_ENABLED

  LOG("Initialized Kitsune runtime (serial)");
}

/// Finalize Kitsune's serial runtime.
extern "C" void __kitser_finalize(void) {
  if (!__kitser_initialized()) {
    LOG("Cannot finalize runtime. Not initialized");
    return;
  }

  LOG("Finalizing Kitsune runtime (serial)");

#ifdef KITRT_PAPI_ENABLED
  __kitpapi_finalize();
#endif // KITRT_PAPI_ENABLED
  KitSerContext::delSingleton();
  LOG("Finalized Kitsune runtime (serial)");

  __kitrt_finalize();
}
