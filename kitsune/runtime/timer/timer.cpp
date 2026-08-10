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
// compiled with Kitsune. Other generic instrumentation interfaces, such as
// LLVM's Xray, typically work at the level of functions, we are interested in
// adding instrumentation for Kitsune's language constructs. We also need to be
// able to time both within and across threads. Having something customized
// should make it easier to use and maintain. This is not intended to be
// production-quality, but good enough to allow us to perform experiments
// quickly.
//
//===----------------------------------------------------------------------===//

#include "timer/timer.h"
#include "common/env.h"
#include "common/logging.h"
#include "global/global.h"
#include "timer/context.h"

#include <ctime>

using namespace kitrt;

/// A time point. This is generally the wallclock time, in nanoseconds, since
/// the epoch.
using TimePoint = TimeSpan;

// The number of nanoseconds since the epoch. This uses CLOCK_REALTIME to get
// wall-clock time. This is susceptible to changes to the system time. This is
// not a situation that we need to defend against.
static TimePoint nsecs() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

TimerEpoch::TimerEpoch(const char *name, KitThreadID thrd)
    : EpochBase(name, thrd) {}

void TimerEpoch::start() { span -= nsecs(); }

TimeSpan TimerEpoch::stop() { return span += nsecs(); }

void TimerEpoch::writeJSON(FILE *fp) const { fprintf(fp, "\n      %ld", span); }

void TimerContext::initialize() {
  // Nothing to be done here.
}

void TimerContext::finalize() { writeJSON(envTimingFile); }

TimerEpoch *TimerContext::makeEpoch(const char *name, KitThreadID thrd) {
  return new TimerEpoch(name, thrd);
}

// -----------------------------------------------------------------------------
// Everything below this is the public interface.

extern "C" KitTimerEpoch *__kittimer_start(const char *name, KitThreadID thrd) {
  TimerEpoch *epoch = mutCtx<TimerContext>().addEpoch(name, thrd);
  epoch->start();
  return reinterpret_cast<KitTimerEpoch *>(epoch);
}

extern "C" KitTimeSpan __kittimer_stop(KitTimerEpoch *handle) {
  gctx.ensure<TimerContext>();
  if (auto *epoch = reinterpret_cast<TimerEpoch *>(handle))
    return epoch->stop();
  return 0;
}
