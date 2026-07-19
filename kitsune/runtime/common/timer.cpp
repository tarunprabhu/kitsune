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
#include "global/singleton.h"

#include <algorithm>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace kitrt {

// A class that wraps all the timers created in the application. A singleton
// instance of this class will be created in the global constructor and will
// live till the global destructor is run.
class KitTimerContext : public KitContextMixin<KitTimerContext> {
public:
  // Each timer will record a number of intervals - one for each occasion when
  // the timer was started and stopped.
  using Intervals = std::vector<TimeSpan>;

  // The key for a timer in the timer map is a combination of the timer id and
  // the thread id.
  using TimerKey = std::pair<TimerID, ThreadID>;

  // The timers in a timer context.
  using Timers = std::map<TimerKey, Intervals>;

private:
  // The mutex that guards access to \ref tmap. Depending on how many threads
  // are in flight, tmap may be being modified while another thread tries to
  // read from it - which can have unpleasant consequences.
  std::mutex mtx;

  // The names of the timers. These are used to print the results. The timer
  // name cannot be inferred from the id.
  std::map<TimerID, std::string> names;

  // The actual timers. The key is a pair of the TimerID and ThreadID. Each
  // ID consists of a number of intervals - each of which is the wallclock time
  // that has elapsed between a pair of calls to \ref __kittimer_start, and
  // \ref __kittimer_stop.
  std::map<TimerKey, Intervals> timers;

public:
  void add(TimeSpan span, TimerID timer, ThreadID thrd, const char *name) {
    std::lock_guard<std::mutex> guard(mtx);
    TimerKey key(timer, thrd);
    if (timers.find(key) == timers.end()) {
      names[timer] = name;
      timers.try_emplace(key);
    }
    timers.at(key).push_back(span);
  }

  const Intervals &get(TimerID timer, ThreadID thrd) const {
    // We don't guard this with a lock because this will only ever be called
    // from the global destructor, __kittimer_finalize(), and that is guaranteed
    // to only run from a single thread.
    return timers.at({timer, thrd});
  }

  const std::string &name(TimerID timer) const { return names.at(timer); }
  bool empty() const { return timers.empty(); }
  size_t size() const { return timers.size(); }
  Timers::const_iterator begin() const { return timers.begin(); }
  Timers::const_iterator end() const { return timers.end(); }
};

} // namespace kitrt

using namespace kitrt;

static TimePoint nsecs() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

using ThreadIDs = std::vector<ThreadID>;
using Timers = std::vector<std::tuple<TimerID, ThreadIDs>>;

static void printTimes(FILE *file, const KitTimerContext::Intervals &spans) {
  bool comma = false;

  fprintf(file, "[");
  for (TimeSpan t : spans) {
    if (comma)
      fprintf(file, ", ");
    fprintf(file, "%lu", t);
    comma = true;
  }
  fprintf(file, "]");
}

static void printThreads(const KitTimerContext &timerCtx, FILE *fp,
                         TimerID timerID, const ThreadIDs &thrdIDs) {
  bool comma = false;

  fprintf(fp, "{");
  for (ThreadID threadID : thrdIDs) {
    if (comma)
      fprintf(fp, ",");
    fprintf(fp, "\n");
    fprintf(fp, "    \"%ld\": ", threadID);
    printTimes(fp, timerCtx.get(timerID, threadID));
    comma = true;
  }
  fprintf(fp, "\n  }");
}

static void printTimers(const KitTimerContext &timerCtx, FILE *fp,
                        const Timers &timers) {
  bool comma = false;

  fprintf(fp, "{");
  for (const auto &[timerID, thrdIDs] : timers) {
    if (comma)
      fprintf(fp, ",");
    fprintf(fp, "\n");
    fprintf(fp, "  \"%s\": ", timerCtx.name(timerID).c_str());
    printThreads(timerCtx, fp, timerID, thrdIDs);
    comma = true;
  }
  fprintf(fp, "\n}\n");
}

static FILE *getFile() {
  if (std::optional<std::string> fname = envLookup(envTimingFile)) {
    if (fname == "-")
      return stdout;

    LOG("Writing timings to file: %s", fname->c_str());
    FILE *fp = fopen(fname->c_str(), "wt");
    if (!fp)
      WARN("Could not open file for writing");
    return fp;
  } else {
    return stderr;
  }
}

static void writeTimings(const KitTimerContext &timerCtx) {
  if (timerCtx.empty())
    return;

  // Sort the timers by name. At the end of this, the vector of pairs might look
  // something like this:
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
  for (const auto &[key, _] : timerCtx)
    ordered.emplace_back(key.first, key.second);
  std::sort(ordered.begin(), ordered.end(),
            [&timerCtx](const auto &p1, const auto &p2) -> bool {
              return timerCtx.name(p1.first) < timerCtx.name(p2.first);
            });

  // Collect the thread id's for each timer. At the end of this, the `ids`
  // variable will look like this:
  //
  //    [{57, [0]},
  //     {9,  [43, 12, 27]},
  //     {98, [0]}]
  //
  Timers timers = {{ordered.front().first, {}}};
  for (const auto &[timerID, thrdID] : ordered) {
    if (timerID != std::get<TimerID>(timers.back()))
      timers.emplace_back(timerID, ThreadIDs());
    std::get<ThreadIDs>(timers.back()).push_back(thrdID);
  }

  // Sort the thread IDs, just because.
  for (auto &[timerID, thrdIDs] : timers)
    std::sort(thrdIDs.begin(), thrdIDs.end());

  // If a timing file is not provided, write timings to stderr. If the name of
  // the timings file is "-", write to stdout. Otherwise, try to write to the
  // file.
  //
  // If the file could not be opened, fp will be nullptr.
  if (FILE *fp = getFile()) {
    printTimers(timerCtx, fp, timers);
    if (fp != stdout && fp != stderr) {
      fclose(fp);
      LOG("Timings written to file");
    }
  }
}

extern "C" TimePoint __kittimer_start(void) { return nsecs(); }

extern "C" TimeSpan __kittimer_stop(TimePoint start, TimerID timer,
                                    ThreadID thrd, const char *name) {
  TimeSpan span = nsecs() - start;
  KitTimerContext::mutSingleton().add(span, timer, thrd, name);

  return span;
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

  writeTimings(KitTimerContext::getSingleton());
  KitTimerContext::delSingleton();

  LOG("Finalized Kitsune timing context");
}
