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

#include "papi/kitpapi.h"
#include "common/env.h"
#include "common/logging.h"
#include "global/global.h"
#include "papi/context.h"

#include "papi.h"

#include <cstdarg>
#include <cstring>
#include <optional>
#include <vector>

using namespace kitrt;

#define CHECK(call, msg)                                                       \
  do {                                                                         \
    if (int err = call)                                                        \
      return handleError("Could not " msg, err);                               \
  } while (0)

// Handle an error returned by a call to a PAPI API function. This will print a
// warning message to stderr. \p what is a custom message to print before the
// actual PAPI error. \p err is the error code returned by PAPI.
static void handleError(const char *what, int err) {
  WARN("%s. %s", what, PAPI_strerror(err));
}

static std::optional<PAPI_event_info_t> getEventInfo(PAPIEventID evt) {
  PAPI_event_info_t evtInfo;
  if (int err = PAPI_get_event_info(evt, &evtInfo)) {
    WARN("Could not get event info. %s", PAPI_strerror(err));
    handleError("Could not get event info", err);
    return std::nullopt;
  }
  return evtInfo;
}

static std::vector<PAPIEventID> getEvents(PAPIEventSet evtSet) {
  if (evtSet == PAPI_NULL)
    return {};

  int n = PAPI_num_events(evtSet);
  std::vector<PAPIEventID> evts(n);
  if (int err = PAPI_list_events(evtSet, evts.data(), &n)) {
    handleError("Could not get events in event set", err);
    return {};
  }

  return evts;
}

static std::vector<PAPIEventID> getAllKnownEvents() {
  std::vector<PAPIEventID> evts;

  // It is not clear what or'ing with 0 is meant to signify. This is how it is
  // implemented in papi_avail.c.
  PAPIEventID evt = PAPI_PRESET_MASK | 0;
  if (int err = PAPI_enum_event(&evt, PAPI_ENUM_FIRST)) {
    handleError("Could not get any PAPI presets", err);
    return evts;
  }

  evts.push_back(evt);
  while (PAPI_enum_event(&evt, PAPI_ENUM_EVENTS) == PAPI_OK)
    evts.push_back(evt);

  return evts;
}

PAPIEpoch::PAPIEpoch(const char *name, KitThreadID thrd, PAPIEventSet evtSet)
    : EpochBase(name, thrd), evtSet(evtSet) {
  counters.reset(new PAPICounter[numEvents()]);
  memset(counters.get(), 0, numEvents() * sizeof(PAPICounter));
}

unsigned PAPIEpoch::numEvents() const { return PAPI_num_events(evtSet); }

void PAPIEpoch::start() {
  LOG("Starting PAPI counters for epoch '%s' on thread '%ld'", name(), thrd());

  // Allocate this before we start recording counters. Strictly speaking, this
  // must be done before PAPI_read, it doesn't have to be done before
  // PAPI_start, but doing it here keeps this function closer to a mirror of
  // PAPIEpoch::stop().
  init = new PAPICounter[numEvents()];

  // Calling PAPI_start the first time can be expensive because PAPI has to
  // make calls into the kernel. Unless we make the compiler do some extra
  // work, we can't know if this is the first time PAPI_start is being called.
  // Instead, we call PAPI_start right away, but read the value of the
  // counters immediately after it returns. These values will be subtracted
  // from those obtained when PAPI_stop is called in __kitpapi_stop.
  CHECK(PAPI_start(evtSet), "start PAPI counters");
  CHECK(PAPI_read(evtSet, init), "read initial values of PAPI counters");
}

void PAPIEpoch::stop() {
  CHECK(PAPI_accum(evtSet, counters.get()), "accumulate PAPI counters");
  CHECK(PAPI_stop(evtSet, nullptr), "stop PAPI counters");

  LOG("Stopped PAPI counters for epoch '%s' on thread '%ld'", name(), thrd());

  for (unsigned i = 0, e = numEvents(); i < e; ++i)
    counters[i] -= init[i];
  delete[] init;
}

void PAPIEpoch::writeJSON(FILE *fp) const {
  std::vector<PAPIEventID> evts = getEvents(evtSet);

  fprintf(fp, "\n      {");
  for (unsigned i = 0, e = evts.size(); i < e; ++i) {
    if (i)
      fprintf(fp, ", ");
    if (std::optional<PAPI_event_info_t> evtInfo = getEventInfo(evts[i]))
      fprintf(fp, "\"%s\": %lld", evtInfo->short_descr, counters[i]);
    else
      fprintf(fp, "\"<unknown>\": 0");
  }
  fprintf(fp, "}");
}

PAPIEpoch *PAPIContext::makeEpoch(const char *name, KitThreadID thrd,
                                  uint32_t n, va_list va) {
  EpochID id = {name, thrd};
  auto it = evtSets.find(id);
  if (it != evtSets.end())
    return new PAPIEpoch(name, thrd, it->second);

  PAPIEventSet evtSet = PAPI_NULL;
  if (int err = PAPI_create_eventset(&evtSet)) {
    handleError("Could not create PAPI event set", err);
    evtSets.emplace(id, PAPI_NULL);
    return new PAPIEpoch(name, thrd, PAPI_NULL);
  }

  for (uint32_t i = 0; i < n; ++i) {
    const char *evtName = va_arg(va, const char *);
    auto it = evtNames.find(evtName);
    if (it == evtNames.end()) {
      WARN("Unknown event name '%s'", evtName);
      continue;
    }

    PAPIEventID evt = it->second;
    std::optional<PAPI_event_info_t> evtInfo = getEventInfo(evt);
    if (!evtInfo.has_value())
      continue;

    const char *s = evtInfo->symbol;
    if (!evtInfo->count)
      WARN("Event '%s' not available", s);
    else if (int e = PAPI_add_event(evtSet, evt))
      WARN("Event '%s' not added to epoch '%s'. %s", s, name, PAPI_strerror(e));
    else
      LOG("Event '%s' added to epoch '%s'", s, name);
  }
  evtSets.emplace(id, evtSet);

  return new PAPIEpoch(name, thrd, evtSet);
}

void PAPIContext::initialize(PAPIThreadIDFunc *getThreadID) {
  LOG("Initializing PAPI library");
  if (int rv = PAPI_library_init(PAPI_VER_CURRENT))
    if (rv != PAPI_VER_CURRENT)
      FATAL("Failed to initialize PAPI");

  if (getThreadID) {
    LOG("Initializing PAPI threading support");
    if (int e = PAPI_thread_init(getThreadID))
      FATAL("Failed to initialize PAPI threading support. %s",
            PAPI_strerror(e));
    LOG("Initialized PAPI threading support");
  }
  LOG("Initialized PAPI library");

  static const std::pair<const char *, PAPIEventID> convenienceNames[] = {
      {"l1d", PAPI_L1_DCM},    {"l2d", PAPI_L2_DCM},  {"l3d", PAPI_L3_DCM},
      {"l1i", PAPI_L1_ICM},    {"l2i", PAPI_L2_ICM},  {"l3i", PAPI_L3_ICM},
      {"l1t", PAPI_L1_TCM},    {"l2t", PAPI_L2_TCM},  {"l3t", PAPI_L3_TCM},
      {"l1ld", PAPI_L1_LDM},   {"l2ld", PAPI_L2_LDM}, {"l3ld", PAPI_L3_LDM},
      {"l1st", PAPI_L1_STM},   {"l2st", PAPI_L2_STM}, {"l3st", PAPI_L3_STM},
      {"tlbd", PAPI_TLB_DM},   {"tlbi", PAPI_TLB_IM}, {"tlbt", PAPI_TLB_TL},
      {"inst", PAPI_TOT_INS},  {"ins", PAPI_TOT_INS}, {"vec", PAPI_VEC_INS},
      {"ld", PAPI_LD_INS},     {"st", PAPI_SR_INS},   {"br", PAPI_BR_INS},
      {"int", PAPI_INT_INS},   {"fp", PAPI_FP_INS},   {"fma", PAPI_FMA_INS},
      {"stall", PAPI_RES_STL}, {"cyc", PAPI_TOT_CYC}, {"ref", PAPI_REF_CYC},
  };

  for (const auto &[name, evt] : convenienceNames)
    evtNames[name] = evt;

  for (PAPIEventID evt : getAllKnownEvents()) {
    // Convert the symbol name to lowercase and drop the PAPI_ prefix. For
    // example, convert "PAPI_TOT_INS" to "tot_ins".
    if (std::optional<PAPI_event_info_t> evtInfo = getEventInfo(evt)) {
      std::string s = &evtInfo->symbol[5];
      for (size_t i = 0, e = s.size(); i < e; ++i)
        s[i] = tolower(s[i]);
      evtNames[s] = evt;
    }
  }
}

void PAPIContext::finalize() {
  writeJSON(envPAPIFile);

  for (auto &[_, evtSet] : evtSets) {
    if (int err = PAPI_cleanup_eventset(evtSet))
      handleError("Could not cleanup event set", err);
    if (int err = PAPI_destroy_eventset(&evtSet))
      handleError("Could not destroy event set", err);
  }

  LOG("Finalizing PAPI library");
  PAPI_shutdown();
  LOG("Finalized PAPI library");
}

// -----------------------------------------------------------------------------
// Everything below this is the public interface.

extern "C" KitPAPIEpoch *__kitpapi_start(const char *name, KitThreadID thrd,
                                         uint32_t n, ...) {
  va_list va;
  va_start(va, n);
  PAPIEpoch *epoch = mutCtx<PAPIContext>().addEpoch(name, thrd, n, va);
  va_end(va);

  epoch->start();
  return reinterpret_cast<KitPAPIEpoch *>(epoch);
}

extern "C" void __kitpapi_stop(KitPAPIEpoch *handle) {
  gctx.ensure<PAPIContext>();
  if (auto *epoch = reinterpret_cast<PAPIEpoch *>(handle))
    epoch->stop();
}
