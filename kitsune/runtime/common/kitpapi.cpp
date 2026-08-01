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

#define CHECK(call, msg)                                                       \
  do {                                                                         \
    if (int err = call)                                                        \
      return handleError("Could not " msg, err);                               \
  } while (0)

static std::pair<const char *, PAPIEventID> convenienceNames[] = {
    {"l1d", PAPI_L1_DCM},  {"l2d", PAPI_L2_DCM},    {"l3d", PAPI_L3_DCM},
    {"l1i", PAPI_L1_ICM},  {"l2i", PAPI_L2_ICM},    {"l3i", PAPI_L3_ICM},
    {"l1t", PAPI_L1_TCM},  {"l2t", PAPI_L2_TCM},    {"l3t", PAPI_L3_TCM},
    {"l1ld", PAPI_L1_LDM}, {"l1st", PAPI_L1_STM},   {"l2ld", PAPI_L2_LDM},
    {"l2st", PAPI_L2_STM}, {"tlbd", PAPI_TLB_DM},   {"tlbi", PAPI_TLB_IM},
    {"tlbt", PAPI_TLB_TL}, {"inst", PAPI_TOT_INS},  {"ins", PAPI_TOT_INS},
    {"vec", PAPI_VEC_INS}, {"ld", PAPI_LD_INS},     {"st", PAPI_SR_INS},
    {"br", PAPI_BR_INS},   {"int", PAPI_INT_INS},   {"fp", PAPI_FP_INS},
    {"fma", PAPI_FMA_INS}, {"stall", PAPI_RES_STL}, {"cyc", PAPI_TOT_CYC},
    {"ref", PAPI_REF_CYC},
};

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

static const char *getEventShortDescr(PAPIEventID evt) {
  if (std::optional<PAPI_event_info_t> evtInfo = getEventInfo(evt))
    return evtInfo->short_descr;
  return "<unknown>";
}

static const char *getEventSymbol(PAPIEventID evt) {
  if (std::optional<PAPI_event_info_t> evtInfo = getEventInfo(evt))
    return evtInfo->symbol;
  return "<unknown>";
}

static bool isEventAvailable(PAPIEventID evt) {
  if (std::optional<PAPI_event_info_t> evtInfo = getEventInfo(evt))
    if (evtInfo->count)
      return true;
  return false;
}

static std::vector<PAPIEventID> getEvents(PAPIEventSet evtSet) {
  if (evtSet == PAPI_NULL)
    return {};

  int n = PAPI_num_events(evtSet);
  std::vector<PAPIEventID> evts(n);
  if (int err = PAPI_list_events(evtSet, evts.data(), &n))
    handleError("Could not get events in event set", err);
  return evts;
}

static std::vector<PAPIEventID> getAllKnownEvents() {
  // It is not clear what or'ing with 0 is meant to signify. This is how it is
  // implemented in papi_avail.c.
  PAPIEventID evt = PAPI_PRESET_MASK | 0;
  if (int err = PAPI_enum_event(&evt, PAPI_ENUM_FIRST)) {
    handleError("Could not get any PAPI presets", err);
    return {};
  }

  std::vector<PAPIEventID> evts = {evt};
  while (PAPI_enum_event(&evt, PAPI_ENUM_EVENTS) == PAPI_OK)
    evts.push_back(evt);
  return evts;
}

namespace {

// Information for an epoch. This wraps the name and the list of events that are
// recorded in an epoch. We don't use the PAPI event set because those are
// created and destroyed on demand. An ID is automatically generated for each
// uniquely named epoch.
struct KitPAPIEpochInfo {
  const std::string name;
  const KitThreadID thrd;
  PAPIEventSet evtSet;

  KitPAPIEpochInfo(const char *name, KitThreadID thrd, PAPIEventSet evtSet)
      : name(name), thrd(thrd), evtSet(evtSet) {}

  virtual ~KitPAPIEpochInfo() {
    if (int err = PAPI_cleanup_eventset(evtSet))
      handleError("Could not cleanup event set", err);
    if (int err = PAPI_destroy_eventset(&evtSet))
      handleError("Could not destroy event set", err);
  }
};

// An epoch object. This is created each time __kitpapi_new is called.
class KitPAPIEpochImpl {
private:
  const KitPAPIEpochInfo &info;

  // The initial values of the counters. These are read after PAPI_start is
  // called with the event set associated with this epoch. This is a temporary
  // buffer.
  PAPICounter *init = nullptr;
  PAPICounter *counters = nullptr;

public:
  KitPAPIEpochImpl() = delete;
  KitPAPIEpochImpl(const KitPAPIEpochImpl &) = delete;
  KitPAPIEpochImpl(KitPAPIEpochImpl &&) = delete;
  KitPAPIEpochImpl &operator=(const KitPAPIEpochImpl &) = delete;
  KitPAPIEpochImpl &operator=(KitPAPIEpochImpl &&) = delete;

  KitPAPIEpochImpl(const KitPAPIEpochInfo &info) : info(info) {
    counters = (PAPICounter *)std::calloc(size(), sizeof(PAPICounter));
  }

  ~KitPAPIEpochImpl() { delete counters; }

  inline const char *name() const { return info.name.c_str(); }
  inline KitThreadID thrd() const { return info.thrd; }
  inline unsigned size() const { return PAPI_num_events(info.evtSet); }
  inline PAPIEventSet evtSet() const { return info.evtSet; }
  inline PAPICounter counter(unsigned i) const { return counters[i]; }

  inline void start() {
    LOG("Starting PAPI counters");

    // Allocate this before we start recording counters. Strictly speaking, this
    // must be done before PAPI_read, it doesn't have to be done before
    // PAPI_start, but doing it here keeps this function closer to a mirror of
    // KitPAPIEpochImpl::stop().
    init = new PAPICounter[size()];

    // Calling PAPI_start the first time can be expensive because PAPI has to
    // make calls into the kernel. Unless we make the compiler do some extra
    // work, we can't know if this is the first time PAPI_start is being called.
    // Instead, we call PAPI_start right away, but read the value of the
    // counters immediately after it returns. These values will be subtracted
    // from those obtained when PAPI_stop is called in __kitpapi_stop.
    CHECK(PAPI_start(evtSet()), "start PAPI counters");
    CHECK(PAPI_read(evtSet(), init), "read initial values of PAPI counters");
  }

  inline void stop() {
    CHECK(PAPI_accum(evtSet(), counters), "read final values of PAPI counters");
    CHECK(PAPI_stop(evtSet(), nullptr), "stop PAPI counters");

    LOG("Stopped PAPI counters");

    for (unsigned i = 0, e = size(); i < e; ++i)
      counters[i] -= init[i];
    delete[] init;
  }
};

} // namespace

namespace kitrt {

using KitPAPIContextBase =
    KitInstrBase<KitPAPIContext, KitPAPIEpochImpl, KitPAPIEpochInfo>;

// The global singleton context for all PAPI events in this context.
class KitPAPIContext : public KitPAPIContextBase {
  friend KitPAPIContextBase;

private:
  // The names of PAPI events that are recognized by this context.
  std::map<std::string, PAPIEventID> evtNames;

private:
  void addNameForEvent(PAPIEventID evt) {
    // Convert the symbol name to lowercase and drop the PAPI_ prefix. For
    // example, convert "PAPI_TOT_INS" to "tot_ins".
    std::string s = &getEventSymbol(evt)[5];
    for (size_t i = 0, e = s.size(); i < e; ++i)
      s[i] = tolower(s[i]);
    evtNames[s] = evt;
  }

protected:
  KitPAPIEpochInfo *makeEpochInfo(const char *name, KitThreadID thrd,
                                  uint32_t n, va_list va) const {
    PAPIEventSet evtSet = PAPI_NULL;
    if (int err = PAPI_create_eventset(&evtSet)) {
      handleError("Could not create PAPI event set", err);
      return new KitPAPIEpochInfo(name, thrd, PAPI_NULL);
    }

    for (uint32_t i = 0; i < n; ++i) {
      const char *evtName = va_arg(va, const char *);
      auto it = evtNames.find(evtName);
      PAPIEventID evt = it->second;
      if (it == evtNames.end())
        WARN("Unknown event name '%s'", evtName);
      else if (!isEventAvailable(evt))
        WARN("Event '%s' not available", getEventSymbol(evt));
      else if (int err = PAPI_add_event(evtSet, evt))
        WARN("Not Event '%s' not added to event set. %s", getEventSymbol(evt),
             PAPI_strerror(err));
      else
        LOG("Event '%s' added to epoch '%s'", getEventSymbol(evt), name);
    }
    return new KitPAPIEpochInfo(name, thrd, evtSet);
  }

  void writeEpoch(FILE *fp, const KitPAPIEpochImpl &epoch) const {
    std::vector<PAPIEventID> evts = getEvents(epoch.evtSet());

    fprintf(fp, "\n      {");
    for (unsigned i = 0, e = evts.size(); i < e; ++i) {
      if (i)
        fprintf(fp, ", ");
      fprintf(fp, "\"%s\": %lld", getEventShortDescr(evts[i]),
              epoch.counter(i));
    }
    fprintf(fp, "}");
  }

public:
  KitPAPIContext() {
    for (const auto &[name, evt] : convenienceNames)
      evtNames[name] = evt;
    for (PAPIEventID evt : getAllKnownEvents())
      addNameForEvent(evt);
  }
};

} // namespace kitrt

using namespace kitrt;

// The default thread ID that is used when a thread function is not provided
// to __kitpapi_initialize(). Always returns 0.
static unsigned long getDefaultThreadID(void) { return 0; }

extern "C" KitPAPIEpoch *__kitpapi_start(const char *name, KitThreadID thrd,
                                         uint32_t n, ...) {
  va_list va;
  va_start(va, n);
  KitPAPIEpochImpl *epoch =
      KitPAPIContext::mutSingleton().addEpoch(name, thrd, n, va);
  va_end(va);
  epoch->start();
  return reinterpret_cast<KitPAPIEpoch *>(epoch);
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
