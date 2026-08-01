//===- timer.cpp - Utilities to collect timings ---------------------------===//
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
// This is intended to be a convenient interface to collect times in code
// compiled with Kitsune. While other generic instrumentation interfaces, such
// as LLVM's Xray, these typically work at the level of functions, while we
// are interested in adding instrumentation for Kitsune's language constructs.
// We also need to be able to time both within and across threads. Having
// something customized should make it easier to use and maintain. This is not
// intended to be production-quality, but good enough to allow us to perform
// experiments quickly.
//
//===----------------------------------------------------------------------===//

#include "common/timer.h"
#include "common/env.h"
#include "common/instrbase.h"
#include "common/logging.h"
#include "global/singleton.h"

#include <ctime>

namespace {

// A time point. This is usually the number of nanoseconds since the epoch.
using KitTimePoint = uint64_t;

// The number of nanoseconds since the epoch. This uses CLOCK_REALTIME to get
// wall-clock time. This is susceptible to changes to the system time. This is
// not a situation that we need to defend against.
static KitTimePoint nsecs() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

// The information unique to an epoch.
struct KitTimerEpochInfo {
  const std::string name;
};

class KitTimerEpochImpl {
public:
  const KitTimerEpochInfo &info;
  const KitThreadID thrd_;
  int64_t span_;

public:
  KitTimerEpochImpl() = delete;
  KitTimerEpochImpl(const KitTimerEpoch &) = delete;
  KitTimerEpochImpl(KitTimerEpoch &&) = delete;
  KitTimerEpochImpl &operator=(const KitTimerEpoch &) = delete;
  KitTimerEpochImpl &operator=(KitTimerEpoch &&) = delete;

  KitTimerEpochImpl(const KitTimerEpochInfo &info, KitThreadID thrd)
      : info(info), thrd_(thrd), span_(0) {}

  inline const std::string &name() const { return info.name; }
  inline KitThreadID thrd() const { return thrd_; }
  inline KitTimeSpan span() const { return span_; }

  inline void start() { span_ -= nsecs(); }

  inline KitTimeSpan stop() {
    span_ += nsecs();
    return span_;
  }
};

} // namespace

namespace kitrt {

using KitTimerContextBase =
    KitInstrBase<KitTimerContext, KitTimerEpochImpl, KitTimerEpochInfo>;

// A class that wraps all the timers created in the application. A singleton
// instance of this class will be created in the global constructor and will
// live till the global destructor is run.
class KitTimerContext : public KitTimerContextBase {
  friend KitTimerContextBase;

protected:
  KitTimerEpochImpl *makeEpoch(const KitTimerEpochInfo &info,
                               KitThreadID thrd) const {
    return new KitTimerEpochImpl(info, thrd);
  }

  KitTimerEpochInfo *makeEpochInfo(const std::string &name) const {
    return new KitTimerEpochInfo{name};
  }

  void writeEpoch(FILE *fp, const KitTimerEpochImpl &epoch) const {
    fprintf(fp, "\n      %ld", epoch.span());
  }
};

} // namespace kitrt

using namespace kitrt;

extern "C" KitTimerEpoch *__kittimer_start(const char *name, KitThreadID thrd) {
  KitTimerContext &ctx = KitTimerContext::mutSingleton();
  const KitTimerEpochInfo &info = ctx.registerEpoch(name);
  KitTimerEpochImpl *epoch = ctx.addEpoch(info, thrd);

  epoch->start();
  return reinterpret_cast<KitTimerEpoch *>(epoch);
}

extern "C" KitTimeSpan __kittimer_stop(KitTimerEpoch *handle) {
  if (KitTimerEpochImpl *epoch = reinterpret_cast<KitTimerEpochImpl *>(handle))
    return epoch->stop();
  return 0;
}

extern "C" bool __kittimer_initialized(void) {
  return KitTimerContext::hasSingleton();
}

extern "C" void __kittimer_initialize(void) {
  if (__kittimer_initialized()) {
    LOG("Timing context already initialized");
    return;
  }

  LOG("Initializing Kitsune timing context");

  KitTimerContext::addSingleton();

  LOG("Initialized Kitsune timing context");
}

extern "C" void __kittimer_finalize(void) {
  if (!__kittimer_initialized()) {
    LOG("Cannot finalize timing context. Not initialized");
    return;
  }

  LOG("Finalizing Kitsune timing context");

  KitTimerContext::getSingleton().writeJSON(envTimingFile);
  KitTimerContext::delSingleton();

  LOG("Finalized Kitsune timing context");
}
