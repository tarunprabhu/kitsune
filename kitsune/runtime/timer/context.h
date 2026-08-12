//===- timer/context.h - Context for the timing support runtime -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Context object for any state required by the Kitsune's timing support
// runtime.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_TIMER_CONTEXT_H
#define KITRT_TIMER_CONTEXT_H

#include "common/instr.h"

#include <cstdint>

namespace kitrt {

/// A time span. This is expected to be the wallclock time, in nanoseconds, that
/// have elapsed between a pair of calls to \ref __kittimer_start and
/// \ref __kittimer_stop.
using TimeSpan = uint64_t;

/// A time point. This is generally the wallclock time, in nanoseconds, since
/// the epoch.
using TimePoint = uint64_t;

/// A timer epoch is a Single-Entry-Single-Exit (SESE) region of code, bounded
/// by calls to \ref __kittimer_start and \ref __kittimer_stop, whose execution
/// time is recorded. The epoch is given a name that ought to be meaningful to
/// the user. An epoch may be started on each thread in a multi-threaded
/// context, or on the main thread of a single-threaded program.
///
/// This struct represents a single epoch. An instance is created by
/// \ref __kittimer_start. It can be used, exactly once, to stop the timer.
class TimerEpoch : public EpochBase {
private:
  /// This is set to a time point when \ref start is called. Its value should
  /// only be read in \ref stop and nowhere else.
  TimePoint tick = 0;

  /// The total number of times that this epoch has been visited. This will be
  /// incremented each time \ref stop is called.
  uint64_t visits = 0;

  /// The total time spent, across visits, spent in this epoch.
  TimeSpan total = 0;

  /// The minimum time, across visits, spent in this epoch.
  TimeSpan min = 0;

  /// The maximum time, across visits, spent in this epoch.
  TimeSpan max = 0;

public:
  TimerEpoch() = delete;
  TimerEpoch(const TimerEpoch &) = delete;
  TimerEpoch(TimerEpoch &&) = delete;
  TimerEpoch &operator=(const TimerEpoch &) = delete;
  TimerEpoch &operator=(TimerEpoch &&) = delete;

  TimerEpoch(const char *name, KitThreadID thrd);

  /// Start recording time for this epoch.
  void start();

  /// Stop recording time for this epoch. Returns the time spent in the epoch
  /// during the current visit.
  TimeSpan stop();
  void writeJSON(FILE *fp) const;
};

// A class that wraps all the timers created in the application. A singleton
// instance of this class will be created in the global constructor and will
// live till the global destructor is run.
class TimerContext : public InstrBase<TimerContext, TimerEpoch> {
  friend InstrBase<TimerContext, TimerEpoch>;

protected:
  TimerEpoch *makeEpoch(const char *name, KitThreadID thrd);

public:
  void initialize();
  void finalize();
};

} // namespace kitrt

#endif // KITRT_TIMER_CONTEXT_H
