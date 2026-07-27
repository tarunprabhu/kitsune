//===- kitpapi.cpp - Kitsune runtime interface for PAPI -------------------===//
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
// These utilities are a convenience wrapper to allow us to add quick
// instrumentation to record hardware counters using PAPI.
//
//===----------------------------------------------------------------------===//

#include "kitpapi.h"
#include "common/env.h"
#include "common/instrbase.h"
#include "common/logging.h"
#include "global/singleton.h"

#include "papi.h"

#include <cstdarg>
#include <optional>
#include <sstream>
#include <vector>

using PAPICounter = long long;
using PAPIEventID = int;
using PAPIEventSet = int;

template <typename T> static std::string to_hex_string(T n) {
  std::stringstream ss;
  ss << "0x" << std::hex << n;
  return ss.str();
}

// Handle an error returned by a call to a PAPI API function. This will print a
// warning message to stderr. \p what is an optional label to print before
// printing the actual PAPI error message. \p err is the error code returned by
// PAPI.
static void handleError(const char *what, int err) {
  WARN("%s. %s", what, PAPI_strerror(err));
}

static std::optional<PAPI_event_info_t> getEventInfo(PAPIEventID evt) {
  PAPI_event_info_t evtInfo;
  if (int err = PAPI_get_event_info(evt, &evtInfo)) {
    handleError("Could not get event info", err);
    return std::nullopt;
  }
  return evtInfo;
}

static std::string getEventShortDescr(PAPIEventID evt) {
  if (std::optional<PAPI_event_info_t> evtInfo = getEventInfo(evt))
    return evtInfo->short_descr;
  return to_hex_string(evt);
}

static std::string getEventSymbol(PAPIEventID evt) {
  if (std::optional<PAPI_event_info_t> evtInfo = getEventInfo(evt))
    return evtInfo->symbol;
  return to_hex_string(evt);
}

namespace {

// Information for an epoch. This wraps the name and the list of events that are
// recorded in an epoch. We don't use the PAPI event set because those are
// created and destroyed on demand. An ID is automatically generated for each
// uniquely named epoch.
struct KitPAPIEpochInfo {
  const std::string name;
  const std::vector<PAPIEventID> evts;
};

// An epoch object. This is created each time __kitpapi_new is called.
class KitPAPIEpochImpl {
private:
  const KitPAPIEpochInfo &info;

  // The ID of the thread on which this epoch will measure counters.
  PAPIThreadID thrd_;

  // The event set for this epoch.
  PAPIEventSet evtSet = PAPI_NULL;

  // The initial values of the counters. These are read after PAPI_start is
  // called with the event set associated with this epoch.
  PAPICounter *initValues = nullptr;

  // The final values of the counters. These are read by __kitpapi_stop. After
  // __kitpapi_stop returns, the corresponding value in \ref initValues will be
  // subtracted from this.
  PAPICounter *finalValues = nullptr;

public:
  KitPAPIEpochImpl() = delete;
  KitPAPIEpochImpl(const KitPAPIEpochImpl &) = delete;
  KitPAPIEpochImpl(KitPAPIEpochImpl &&) = delete;
  KitPAPIEpochImpl &operator=(const KitPAPIEpochImpl &) = delete;
  KitPAPIEpochImpl &operator=(KitPAPIEpochImpl &&) = delete;

  KitPAPIEpochImpl(const KitPAPIEpochInfo &info, PAPIThreadID thrd,
                   PAPIEventSet evtSet)
      : info(info), thrd_(thrd), evtSet(evtSet) {
    int n = size();
    initValues = (PAPICounter *)std::calloc(n, sizeof(PAPICounter));
    finalValues = (PAPICounter *)std::calloc(n, sizeof(PAPICounter));
  }

  ~KitPAPIEpochImpl() { delete finalValues; }

  inline const std::string &name() const { return info.name; }
  inline PAPIThreadID thrd() const { return thrd_; }
  inline unsigned size() const { return info.evts.size(); }
  inline PAPIEventID event(unsigned i) const { return info.evts[i]; }
  inline PAPICounter value(unsigned i) const { return finalValues[i]; }

  inline void start() {
    LOG("Starting PAPI counters");

    // Calling PAPI_start the first time can be expensive because PAPI has to
    // make calls into the kernel. Unless we make the compiler do some extra
    // work, we can't know if this is the first time PAPI_start is being called.
    // Instead, we call PAPI_start right away, but read the value of the
    // counters immediately after it returns. These values will be subtracted
    // from those obtained when PAPI_stop is called in __kitpapi_stop.
    if (int err = PAPI_start(evtSet))
      return handleError("Could not start PAPI counters", err);

    if (int err = PAPI_read(evtSet, initValues))
      return handleError("Could not read initial values of PAPI counters", err);
  }

  inline void stop() {
    LOG("Stopping PAPI counters");

    if (int err = PAPI_stop(evtSet, finalValues))
      return handleError("Could not read final values of PAPI counters", err);
    for (unsigned i = 0, e = size(); i < e; ++i)
      finalValues[i] -= initValues[i];

    if (int err = PAPI_cleanup_eventset(evtSet))
      handleError("Could not cleanup event set", err);
    if (int err = PAPI_destroy_eventset(&evtSet))
      handleError("Could not destroy event set", err);
    evtSet = PAPI_NULL;

    delete initValues;
  }
};

} // namespace

namespace kitrt {

using KitPAPIContextBase = KitInstrBase<KitPAPIContext, KitPAPIEpochImpl,
                                        KitPAPIEpochInfo, PAPIThreadID>;

// The global singleton context for all PAPI events in this context.
class KitPAPIContext : public KitPAPIContextBase {
  friend KitPAPIContextBase;

protected:
  static KitPAPIEpochInfo *makeEpochInfo(const std::string &name,
                                         const std::vector<PAPIEventID> &evts) {
    return new KitPAPIEpochInfo{name, evts};
  }

  static PAPIEventSet createEventSet(const KitPAPIEpochInfo &info) {
    PAPIEventSet evtSet = PAPI_NULL;
    if (int err = PAPI_create_eventset(&evtSet)) {
      handleError("Could not create PAPI event set", err);
      return PAPI_NULL;
    }

    for (PAPIEventID evt : info.evts)
      if (int err = PAPI_add_event(evtSet, evt))
        WARN("Could not add event '%s'. %s", getEventSymbol(evt).c_str(),
             PAPI_strerror(err));
    return evtSet;
  }

  static KitPAPIEpochImpl *makeEpoch(const KitPAPIEpochInfo &info) {
    PAPIThreadID thrd = PAPI_thread_id();
    PAPIEventSet evtSet = createEventSet(info);
    return new KitPAPIEpochImpl(info, thrd, evtSet);
  }

  static void writeEpoch(FILE *fp, const KitPAPIEpochImpl &epoch) {
    fprintf(fp, "\n      {");
    for (unsigned i = 0, e = epoch.size(); i < e; ++i) {
      if (i)
        fprintf(fp, ", ");
      std::string descr = getEventShortDescr(epoch.event(i));
      fprintf(fp, "\"%s\": %lld", descr.c_str(), epoch.value(i));
    }
    fprintf(fp, "}");
  }
};

} // namespace kitrt

using namespace kitrt;

// The default thread ID that is used when a thread function is not provided
// to __kitpapi_initialize(). Always returns 0.
static unsigned long getDefaultThreadID(void) { return 0; }

extern "C" KitPAPIEpoch *__kitpapi_new(const char *name, ...) {
  assert(name && "Name of a PAPI epoch must not be NULL");

  std::vector<PAPIEventID> evts;
  va_list va;
  va_start(va, name);
  while (PAPIEventID evt = va_arg(va, PAPIEventID))
    evts.push_back(evt);
  va_end(va);

  KitPAPIContext &ctx = KitPAPIContext::mutSingleton();
  const KitPAPIEpochInfo &info = ctx.registerEpoch(name, evts);
  KitPAPIEpochImpl *epoch = ctx.addEpoch(info);

  return reinterpret_cast<KitPAPIEpoch *>(epoch);
}

extern "C" void __kitpapi_start(KitPAPIEpoch *handle) {
  if (KitPAPIEpochImpl *epoch = reinterpret_cast<KitPAPIEpochImpl *>(handle))
    epoch->start();
}

extern "C" void __kitpapi_stop(KitPAPIEpoch *handle) {
  if (KitPAPIEpochImpl *epoch = reinterpret_cast<KitPAPIEpochImpl *>(handle))
    epoch->stop();
}

extern "C" bool __kitpapi_initialized(void) {
  return KitPAPIContext::hasSingleton();
}

extern "C" void __kitpapi_initialize(PAPIThreadIDFunc getThreadID) {
  auto getThreadIDFuncOrDefault = [](PAPIThreadIDFunc f) -> PAPIThreadIDFunc {
    if (f)
      return f;
    return getDefaultThreadID;
  };

  if (__kitpapi_initialized()) {
    LOG("PAPI library already initialized");
    return;
  }

  LOG("Initializing PAPI library");
  if (int rv = PAPI_library_init(PAPI_VER_CURRENT))
    if (rv != PAPI_VER_CURRENT)
      return handleError("Could not initialize PAPI", rv);

  // Only create the singleton instance if the PAPI library was initialized.
  KitPAPIContext::addSingleton();

  LOG("Initializing PAPI threading support");
  if (int e = PAPI_thread_init(getThreadIDFuncOrDefault(getThreadID)))
    handleError("Could not initialize PAPI threading support", e);
  else
    LOG("Initialized PAPI threading support");

  LOG("Initialized PAPI library");
}

extern "C" void __kitpapi_finalize(void) {
  if (!__kitpapi_initialized()) {
    LOG("Cannot finalize PAPI library. Not initialized");
    return;
  }

  LOG("Finalizing PAPI library");

  KitPAPIContext::getSingleton().writeJSON(envPAPIFile);
  KitPAPIContext::delSingleton();
  PAPI_shutdown();

  LOG("Finalized PAPI library");
}
