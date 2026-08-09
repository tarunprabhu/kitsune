//===- timer.h - Utilities to collect timings ------------------*- C++ -*--===//
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

#ifndef KITRT_TIMER_TIMER_H
#define KITRT_TIMER_TIMER_H

#include "common/instrbase.h"
#include "common/thread.h"

#include <cstdint>

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else // !__cplusplus
#define EXTERN_C
#endif // !__cplusplus

namespace kitrt {

/// A time span. This is expected to be the wallclock time, in nanoseconds, that
/// have elapsed between a pair of calls to \ref __kittimer_start and
/// \ref __kittimer_stop.
using KitTimeSpan = int64_t;

/// A timer epoch is a Single-Entry-Single-Exit (SESE) region of code, bounded
/// by calls to \ref __kittimer_start and \ref __kittimer_stop, whose execution
/// time is recorded. The epoch is given a name that ought to be meaningful to
/// the user. An epoch may be started on each thread in a multi-threaded
/// context, or on the main thread of a single-threaded program.
///
/// This struct represents a single epoch. An instance is created by
/// \ref __kittimer_start. It can be used, exactly once, to stop the timer.
class KitTimerEpoch : public KitEpochBase {
private:
  KitTimeSpan span = 0;

public:
  KitTimerEpoch() = delete;
  KitTimerEpoch(const KitTimerEpoch &) = delete;
  KitTimerEpoch(KitTimerEpoch &&) = delete;
  KitTimerEpoch &operator=(const KitTimerEpoch &) = delete;
  KitTimerEpoch &operator=(KitTimerEpoch &&) = delete;

  KitTimerEpoch(const char *name, KitThreadID thrd);

  void start();
  KitTimeSpan stop();
  void writeJSON(FILE *fp) const;
};

using KitTimerContextBase = KitInstrBase<KitTimerContext, KitTimerEpoch>;

// A class that wraps all the timers created in the application. A singleton
// instance of this class will be created in the global constructor and will
// live till the global destructor is run.
class KitTimerContext : public KitTimerContextBase {
  friend KitTimerContextBase;

protected:
  KitTimerEpoch *makeEpoch(const char *name, KitThreadID thrd);

public:
  void initialize();
  void finalize();

public:
  static const char *name() { return "timer"; }
};

} // namespace kitrt

/// Start the timer \p timer. \p thrd is the ID of the thread on which the
/// timer is running. \p name is the name of the timer.
EXTERN_C kitrt::KitTimerEpoch *__kittimer_start(const char *name,
                                                KitThreadID span);

/// Stop a timer running on a thread with ID \p thrd. The runtime
/// will create a mapping between \p timer and \p name, but only if \p timer was
/// not used in an earlier call to this function. Multiple threads can
/// share the a timer ID. This is useful when measuring the times for individual
/// threads launched, for instance, when a parallel loop is lowered using a
/// CPU-centric tapir target. \p start is the TimePoint obtained by calling
/// \ref __kittimer_start. Returns the wallclock time, in nanoseconds, that have
/// elapsed between the time this is called, and the time that \p start was
/// recorded.
EXTERN_C kitrt::KitTimeSpan __kittimer_stop(kitrt::KitTimerEpoch *handle);

#undef EXTERN_C
#endif // KITRT_COMMON_TIMER_H
