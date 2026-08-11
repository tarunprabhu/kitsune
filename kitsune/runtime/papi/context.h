//===- papi/context.h - Context for the PAPI support runtime ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Context object for any state required by the Kitsune's PAPI support runtime.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_PAPI_CONTEXT_H
#define KITRT_PAPI_CONTEXT_H

#include "common/instr.h"

#include <map>
#include <memory>

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
class PAPIEpoch : public EpochBase {
private:
  PAPIEventSet evtSet;

  // The initial values of the counters. These are read after PAPI_start is
  // called. This is a temporary buffer and will be allocated in
  // PAPIEpoch::start() and freed in PAPIEpoch::stop().
  PAPICounter *init = nullptr;

  std::unique_ptr<PAPICounter[]> counters;

private:
  unsigned numEvents() const;

public:
  PAPIEpoch() = delete;
  PAPIEpoch(const PAPIEpoch &) = delete;
  PAPIEpoch(PAPIEpoch &&) = delete;
  PAPIEpoch &operator=(const PAPIEpoch &) = delete;
  PAPIEpoch &operator=(PAPIEpoch &&) = delete;

  PAPIEpoch(const char *name, KitThreadID thrd, PAPIEventSet evtSet);

  void start();
  void stop();
  void writeJSON(FILE *fp) const;
};

// The global singleton context for all PAPI events in this context.
class PAPIContext : public InstrBase<PAPIContext, PAPIEpoch> {
  friend InstrBase<PAPIContext, PAPIEpoch>;

private:
  // The names of PAPI events that are recognized by this context.
  std::map<std::string, PAPIEventID> evtNames;
  std::map<EpochID, PAPIEventSet> evtSets;

protected:
  PAPIEpoch *makeEpoch(const char *name, KitThreadID thrd, uint32_t n,
                       va_list va);

public:
  // Initialize PAPI. If PAPI is to be used in a multi-threaded context, provide
  // a function to get the ID of the thread from which the function was called.
  // This may be nullptr, in which case, PAPI is assumed to be running in a
  // single-threaded context.
  void initialize(PAPIThreadIDFunc *getThreadID);
  void finalize();
};

} // namespace kitrt

#endif // KITRT_PAPI_CONTEXT_H
