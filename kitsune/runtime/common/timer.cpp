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
#include "common/logging.h"

#include <algorithm>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#define LABEL "kitrt"

using namespace kitrt;

namespace {

using TimeSpan = uint64_t;

// A single timer. This can be started and stopped multiple times. Each such
// interval will be recorded separately.
class Timer {
private:
  using Intervals = std::vector<TimeSpan>;

private:
  // All intervals that this timer has measured. An interval is started when
  // the tick() method is called, and stopped when the tock() method is called.
  // Each entry in this array will be the wall-clock time, in nanoseconds,
  // between calls to tick() and tock().
  Intervals intervals;

  // A time-point that is set when the tick() method is called. This is usually
  // the number of nanoseconds since the epoch, but could be something else. A
  // non-zero value indicates that an interval is being measured.
  TimeSpan tick0 = 0;

private:
  TimeSpan nsecs() const {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000000000 + ts.tv_nsec;
  }

public:
  size_t size() const { return intervals.size(); }
  Intervals::const_iterator begin() const { return intervals.begin(); }
  Intervals::const_iterator end() const { return intervals.end(); }

  void tick() { tick0 = nsecs(); }

  void tock() {
    intervals.push_back(nsecs() - tick0);
    tick0 = 0;
  }
};

// A class that wraps all the timers created in the application. A singleton
// instance of this class will be created in the global constructor and will
// live till the global destructor is run.
class KitTimerContext {
public:
  // The key for a timer in the timer map is a combination of the timer id and
  // the thread id.
  using TimerKey = std::pair<TimerID, ThreadID>;

  // The timers in a timer context.
  using Timers = std::map<TimerKey, Timer>;

private:
  // The mutex that guards access to \ref tmap. Depending on how many threads
  // are in flight, tmap may be being modified while another thread tries to
  // read from it - which can have unpleasant consequences.
  std::mutex mtx;

  // The names of the timers. These are used to print the results. The timer
  // name cannot be inferred from the id.
  std::map<TimerID, std::string> tnames;

  // The actual timers. A timer name does not necessarily include the id of the
  // thread on which it is running, so these must be kept separate. It would
  // be good to have a vector instead of a map since the thread id's are likely
  // to be sequential and zero-indexed, but we don't know ahead of time how
  // many there will be, and in what order they will be seen. This would
  // complicate the logic of registering timers.
  std::map<TimerKey, Timer> tmap;

public:
  Timer &get(TimerID timer, ThreadID thrd, const char *name) {
    TimerKey key(timer, thrd);
    if (tmap.find(key) == tmap.end()) {
      std::lock_guard<std::mutex> guard(mtx);
      tnames[timer] = name;
      tmap.try_emplace(key);
    }
    return tmap.at(key);
  }

  Timer &get(TimerID timer, ThreadID thrd) {
    std::lock_guard<std::mutex> guard(mtx);
    return tmap.at({timer, thrd});
  }

  const Timer &get(TimerID timer, ThreadID thrd) const {
    // We don't guard this with a lock because this will only ever be called
    // from the global destructor, __kittimer_finalize(), and that is guaranteed
    // to only run from a single thread.
    return tmap.at({timer, thrd});
  }

  const std::string &name(TimerID timer) const { return tnames.at(timer); }
  bool empty() const { return tmap.empty(); }
  size_t size() const { return tmap.size(); }
  Timers::const_iterator begin() const { return tmap.begin(); }
  Timers::const_iterator end() const { return tmap.end(); }
};

} // namespace

// FIXME: This should be collected into a singleton struct for the entire
// runtime, not just the timing-specific parts of it.
//
// Singleton global timer context for the whole application. This is *NOT*
// intended to be thread-local.
static KitTimerContext *gTimerCtx = nullptr;

extern "C" void __kittimer_start(TimerID timer, ThreadID thrd,
                                 const char *name) {
  gTimerCtx->get(timer, thrd, name).tick();
}

extern "C" void __kittimer_stop(TimerID timer, ThreadID thrd) {
  gTimerCtx->get(timer, thrd).tock();
}

extern "C" void __kittimer_initialize(void) { gTimerCtx = new KitTimerContext; }

extern "C" void __kittimer_finalize(void) {
  using ThreadIDs = std::vector<ThreadID>;
  using IDs = std::vector<std::tuple<TimerID, ThreadIDs>>;

  auto printTimes = [](FILE *file, const Timer &timer) {
    bool comma = false;

    fprintf(file, "[");
    for (TimeSpan t : timer) {
      if (comma)
        fprintf(file, ", ");
      fprintf(file, "%lu", t);
      comma = true;
    }
    fprintf(file, "]");
  };

  auto printThreads = [&](FILE *file, TimerID timerID, ThreadIDs &thrdIDs) {
    std::sort(thrdIDs.begin(), thrdIDs.end());
    bool comma = false;

    fprintf(file, "{");
    for (ThreadID threadID : thrdIDs) {
      if (comma)
        fprintf(file, ",");
      fprintf(file, "\n");
      fprintf(file, "    \"%ld\": ", threadID);
      printTimes(file, gTimerCtx->get(timerID, threadID));
      comma = true;
    }
    fprintf(file, "\n  }");
  };

  auto printTimers = [&](FILE *file, IDs &ids) {
    bool comma = false;

    fprintf(file, "{");
    for (auto &[timerID, thrdIDs] : ids) {
      if (comma)
        fprintf(file, ",");
      fprintf(file, "\n");
      fprintf(file, "  \"%s\": ", gTimerCtx->name(timerID).c_str());
      printThreads(file, timerID, thrdIDs);
      comma = true;
    }
    fprintf(file, "\n}\n");
  };

  auto getFile = []() -> FILE * {
    if (std::optional<std::string> fname = envLookup("KIT_TIMING_FILE")) {
      if (fname == "-")
        return stdout;

      log(LABEL, "Writing timings to file: %s", fname->c_str());
      FILE *fp = fopen(fname->c_str(), "wt");
      if (!fp)
        warn(LABEL, "Could not open file for writing");
      return fp;
    } else {
      return stderr;
    }
  };

  if (gTimerCtx->size()) {
    // Sort the timers by name. At the end of this, the vector of pairs might
    // look something like this:
    //
    //     [{57, 0},   // name = "main"
    //      {9,  43},  // name = "timer1"
    //      {9,  12},  // name = "timer1"
    //      {9,  27},  // name = "timer1"
    //      {98, 0}]   // name = "write"
    //
    // Note that the timers are sorted by their name, not the IDs. However, the
    // threads within each timer may not be sorted.
    //
    std::vector<std::pair<TimerID, ThreadID>> ordered;
    for (const auto &[key, _] : *gTimerCtx)
      ordered.emplace_back(key.first, key.second);
    std::sort(ordered.begin(), ordered.end(),
              [](const auto &p1, const auto &p2) -> bool {
                return gTimerCtx->name(p1.first) < gTimerCtx->name(p2.first);
              });

    // Collect the thread id's for each timer. At the end of this, the `ids`
    // variable will look like this:
    //
    //    [{57, [0]},
    //     {9,  [43, 12, 27]},
    //     {98, [0]}]
    //
    IDs ids = {{ordered.front().first, {}}};
    for (const auto &[timerID, thrdID] : ordered) {
      if (timerID != std::get<TimerID>(ids.back()))
        ids.emplace_back(timerID, ThreadIDs());
      std::get<ThreadIDs>(ids.back()).push_back(thrdID);
    }

    // If a timing file is not provided, write timings to stderr. If the name of
    // the timings file is "-", write to stdout. Otherwise, try to write to the
    // file.
    //
    // If the file could not be opened, fp will be nullptr.
    if (FILE *fp = getFile()) {
      printTimers(fp, ids);
      if (fp != stdout && fp != stderr) {
        fclose(fp);
        log(LABEL, "Timings written to file");
      }
    }
  }

  delete gTimerCtx;
}
