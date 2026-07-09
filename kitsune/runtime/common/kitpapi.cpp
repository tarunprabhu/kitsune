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
#include "common/logging.h"

#include <papi.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#define LABEL "kitpapi"

// A PAPI event context. This contains an optional that can be useful to
// identify the source of the counters when they are printed, and an event set
// created by a call to PAPI_create_event_set. This is intentionally opaque to
// callers. A pointer to this will be used by most of the functions here.
struct KitPAPIContext {
  std::string name = "<anon>";

  /// The event set created by PAPI_create_event_set.
  int evtSet = PAPI_NULL;

  /// The events added to the event set.
  std::vector<int> evts;

  /// The values for the events added to \ref evtSet.
  std::vector<long long> values;

public:
  KitPAPIContext(const char *n, int evtSet) : evtSet(evtSet) {
    if (n && strlen(n) > 0)
      name = n;
  }
};

void __kitpapi_error(const char *what, int err) {
  __kitrt_warn(LABEL, "%s. %s", what, PAPI_strerror(err));
}

static std::string getEventSymbol(int evt) {
  PAPI_event_info_t info;
  int err = PAPI_get_event_info(evt, &info);
  if (!err)
    return info.symbol;
  __kitpapi_error("Could not get event info", err);
  return "<unknown>";
}

static std::string getEventLabel(int evt) {
  PAPI_event_info_t info;
  int err = PAPI_get_event_info(evt, &info);
  if (!err)
    return info.short_descr;
  __kitpapi_error("Could not get event info", err);
  return "<unknown>";
}

static void __kitpapi_add_event_impl(KitPAPIContext *ctx, const char *name,
                                     int evt) {
  std::string evtSymbol = getEventSymbol(evt);
  std::string evtLabel = getEventLabel(evt);
  if (int err = PAPI_add_event(ctx->evtSet, evt))
    return __kitrt_warn(
        LABEL, "Could not add event '%s'. Mapped to %s (%s). %s", name,
        evtSymbol.c_str(), evtLabel.c_str(), PAPI_strerror(err));

  ctx->evts.push_back(evt);
  ctx->values.push_back(0);
  __kitrt_message(LABEL, "Added event '%s'. Mapped to %s (%s)", name,
                  evtSymbol.c_str(), evtLabel.c_str());
}

#define MAYBE_ADD_EVENT(ctx, name, evtName, evtCode)                           \
  do {                                                                         \
    if (strcmp(name, evtName) == 0)                                            \
      return __kitpapi_add_event_impl(ctx, name, evtCode);                     \
  } while (0)

extern "C" void __kitpapi_add_event(KitPAPIContext *ctx, const char *name) {
  if (!ctx)
    return;

  if (!name) {
    __kitrt_error(LABEL, "Event name cannot be null");
    return;
  }

  MAYBE_ADD_EVENT(ctx, name, __kitpapi_l1_dcm, PAPI_L1_DCM);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_l2_dcm, PAPI_L2_DCM);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_l3_dcm, PAPI_L3_DCM);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_l1_icm, PAPI_L1_ICM);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_l2_icm, PAPI_L2_ICM);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_l3_icm, PAPI_L3_ICM);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_l1_tcm, PAPI_L1_TCM);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_l2_tcm, PAPI_L2_TCM);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_l3_tcm, PAPI_L3_TCM);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_tot_inst, PAPI_TOT_INS);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_vec_inst, PAPI_VEC_INS);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_load_inst, PAPI_LD_INS);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_store_inst, PAPI_SR_INS);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_br_inst, PAPI_BR_INS);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_int_inst, PAPI_INT_INS);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_fp_inst, PAPI_FP_INS);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_fma_inst, PAPI_FMA_INS);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_tot_cyc, PAPI_TOT_CYC);
  MAYBE_ADD_EVENT(ctx, name, __kitpapi_ref_cyc, PAPI_REF_CYC);

  __kitrt_warn(LABEL, "Ignoring unknown event '%s'", name);
}

static KitPAPIContext *__kitpapi_new_empty(const char *name) {
  int evtSet = PAPI_NULL;
  int err = PAPI_create_eventset(&evtSet);
  if (!err)
    return new KitPAPIContext(name, evtSet);

  __kitpapi_error("Could not create PAPI event set", err);
  return nullptr;
}

extern "C" KitPAPIContext *__kitpapi_new(const char *name, ...) {
  if (!PAPI_is_initialized())
    return nullptr;

  KitPAPIContext *ctx = __kitpapi_new_empty(name);
  if (!ctx)
    return nullptr;

  va_list args;
  va_start(args, name);
  const char *evt = nullptr;
  do {
    evt = va_arg(args, const char *);
    if (evt)
      __kitpapi_add_event(ctx, evt);
  } while (evt);
  va_end(args);

  return ctx;
}

extern "C" void __kitpapi_start(KitPAPIContext *ctx) {
  if (!ctx)
    return;

  __kitrt_message(LABEL, "Starting PAPI counters");
  if (int err = PAPI_start(ctx->evtSet))
    return __kitpapi_error("Could not start PAPI counters", err);
}

static void __kitpapi_delete(KitPAPIContext *ctx) {
  if (int err = PAPI_cleanup_eventset(ctx->evtSet))
    __kitpapi_error("Could not cleanup event set", err);
  if (int err = PAPI_destroy_eventset(&ctx->evtSet))
    __kitpapi_error("Could not destroy event set", err);
  delete ctx;
}

extern "C" void __kitpapi_stop(KitPAPIContext *ctx) {
  if (!ctx)
    return;

  if (int err = PAPI_stop(ctx->evtSet, ctx->values.data())) {
    __kitpapi_error("Could not stop PAPI counters", err);
    return __kitpapi_delete(ctx);
  }
  __kitrt_message(LABEL, "Stopped PAPI counters");

  static std::mutex mtx;
  std::lock_guard<std::mutex> guard(mtx);
  fprintf(stderr, "PAPI event set '%s' on thread %ld\n", ctx->name.c_str(),
          std::max((long)PAPI_thread_id(), 0L));
  for (unsigned i = 0, numEvts = ctx->evts.size(); i < numEvts; ++i) {
    // We compute the event label out-of-line with the printf because it may
    // print an error message if the event's label could not be computed. In
    // that case, we would still like the output to look somewhat reasonable.
    // If the label could not be computed, a default label will be returned.
    std::string evtLabel = getEventLabel(ctx->evts[i]);
    fprintf(stderr, "  %-32s  %lld\n", evtLabel.c_str(), ctx->values[i]);
  }

  __kitpapi_delete(ctx);
}

extern "C" void __kitpapi_initialize_threading(void *getThreadId) {
  __kitrt_message(LABEL, "Initializing PAPI threading support");
  if (int err = PAPI_thread_init((unsigned long (*)(void))getThreadId))
    return __kitpapi_error("Could not initialize PAPI threading support", err);
  __kitrt_message(LABEL, "Initialized PAPI threading support");
}

extern "C" void __kitpapi_initialize() {
  __kitrt_message(LABEL, "Initializing PAPI library");
  if (int rv = PAPI_library_init(PAPI_VER_CURRENT))
    if (rv != PAPI_VER_CURRENT)
      return __kitpapi_error("Could not initialize PAPI", rv);
  __kitrt_message(LABEL, "Initialized PAPI library");
}

extern "C" void __kitpapi_finalize() {
  __kitrt_message(LABEL, "Finalizing PAPI library");
  PAPI_shutdown();
  __kitrt_message(LABEL, "Finalized PAPI library");
}
