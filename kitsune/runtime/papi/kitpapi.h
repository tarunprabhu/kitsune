//===- kitpapi.h - Kitsune's convenience interface for PAPI ----*- C++ -*--===//
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
// These utilities are a convenience wrapper to allow us to add quick
// instrumentation to record hardware counters using PAPI.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_COMMON_KITPAPI_H
#define KITRT_COMMON_KITPAPI_H

#include "common/instr.h"
#include "common/thread.h"

#include <cstdint>
#include <map>
#include <memory>

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else // !__cplusplus
#define EXTERN_C
#endif // !__cplusplus

namespace kitrt {

/// The type of the function that returns the ID of a thread.
using PAPIThreadIDFunc = unsigned long(void);

using PAPICounter = long long;
using PAPIEventID = int;
using PAPIEventSet = int;

/// A PAPI epoch is a Single-Entry-Single-Exit (SESE) region of code, bounded by
/// calls to \ref __kitpapi_start and \ref __kitpapi_stop, during which PAPI
/// events are recorded. The epoch is given a name that ought to be meaningful
/// to the user. A set of PAPI events is collected during an epoch. An epoch may
/// be started on each thread in a multi-threaded context, or on the main thread
/// of a single-threaded program.
///
/// This struct represents a single epoch. An instance is created by
/// \ref __kitpapi_new. It can be used, exactly once, to start and stop
/// recording events.
class KitPAPIEpoch : public KitEpochBase {
private:
  PAPIEventSet evtSet;

  // The initial values of the counters. These are read after PAPI_start is
  // called. This is a temporary buffer and will be allocated in
  // KitPAPIEpoch::start() and freed in KitPAPIEpoch::stop().
  PAPICounter *init = nullptr;

  std::unique_ptr<PAPICounter[]> counters;

private:
  unsigned numEvents() const;

public:
  KitPAPIEpoch() = delete;
  KitPAPIEpoch(const KitPAPIEpoch &) = delete;
  KitPAPIEpoch(KitPAPIEpoch &&) = delete;
  KitPAPIEpoch &operator=(const KitPAPIEpoch &) = delete;
  KitPAPIEpoch &operator=(KitPAPIEpoch &&) = delete;

  KitPAPIEpoch(const char *name, KitThreadID thrd, PAPIEventSet evtSet);

  void start();
  void stop();
  void writeJSON(FILE *fp) const;
};

using KitPAPIContextBase = KitInstrBase<KitPAPIContext, KitPAPIEpoch>;

// The global singleton context for all PAPI events in this context.
class KitPAPIContext : public KitPAPIContextBase {
  friend KitPAPIContextBase;

private:
  // The names of PAPI events that are recognized by this context.
  std::map<std::string, PAPIEventID> evtNames;
  std::map<EpochID, PAPIEventSet> evtSets;

protected:
  KitPAPIEpoch *makeEpoch(const char *name, KitThreadID thrd, uint32_t n,
                          va_list va);

public:
  // Initialize PAPI. If PAPI is to be used in a multi-threaded context, provide
  // a function to get the ID of the thread from which the function was called.
  // This may be nullptr, in which case, PAPI is assumed to be running in a
  // single-threaded context.
  void initialize(PAPIThreadIDFunc *getThreadID);
  void finalize();

public:
  static const char *name() { return "papi"; }
};

} // namespace kitrt

/// Start a PAPI epoch. \p name must be a globally unique name for the epoch. By
/// globally unique, we mean that it must be guaranteed not to collide with any
/// other identifier that might be encountered during the course of program
/// execution. \p name must not be nullptr. The created epoch must be passed to
/// \ref __kitpapi_start and \ref __kitpapi_stop no more than once. \n is the
/// number of optional arguments passed after \p n. The optional arguments must
/// each be of type `const char*` and must be a recognized name of an event that
/// can be recorded. If any of these are not recognized, or if the events are
/// not available on the system where this is being run, a warning message will
/// be printed.
EXTERN_C kitrt::KitPAPIEpoch *
__kitpapi_start(const char *name, KitThreadID thrd, uint32_t n, ...);

/// Stop collecting events. \p epoch must be an epoch previously created by a
/// __kitpapi_new*.
EXTERN_C void __kitpapi_stop(kitrt::KitPAPIEpoch *epoch);

#undef EXTERN_C
#endif // KITRT_COMMON_KITPAPI_H
