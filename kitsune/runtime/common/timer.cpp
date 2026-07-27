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
#include "common/ptriter.h"
#include "global/singleton.h"

#include <algorithm>
#include <ctime>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

// A time point. This is usually the number of nanoseconds since the epoch.
using KitTimePoint = uint64_t;
using KitTimerID = uint64_t;

static KitTimePoint nsecs() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

struct KitTimerEpochInfo {
  const KitTimerID id;
  const std::string name;

  KitTimerEpochInfo(KitTimerID id, const std::string &name)
      : id(id), name(name) {}
};

class KitTimerEpochImpl {
public:
  const KitTimerEpochInfo &info;
  const KitThreadID thrd_;
  KitTimeSpan span_;

public:
  KitTimerEpochImpl() = delete;
  KitTimerEpochImpl(const KitTimerEpoch &) = delete;
  KitTimerEpochImpl(KitTimerEpoch &&) = delete;
  KitTimerEpochImpl &operator=(const KitTimerEpoch &) = delete;
  KitTimerEpochImpl &operator=(KitTimerEpoch &&) = delete;

  KitTimerEpochImpl(const KitTimerEpochInfo &info, KitThreadID thrd)
      : info(info), thrd_(thrd), span_(0) {}

  inline KitTimerID id() const { return info.id; }
  inline const std::string &name() const { return info.name; }
  inline KitThreadID thrd() const { return thrd_; }
  inline KitTimeSpan span() const { return span_; }

  inline void start() { span_ = nsecs(); }

  inline KitTimeSpan stop() {
    span_ = nsecs() - span_;
    return span_;
  }
};

} // namespace

namespace kitrt {

// A class that wraps all the timers created in the application. A singleton
// instance of this class will be created in the global constructor and will
// live till the global destructor is run.
class KitTimerContext : public KitContextMixin<KitTimerContext> {
private:
  // A mutex that controls all accesses to the mutable members of this class.
  std::mutex mtx;

  // The names of the timers. These are used to print the results. The timer
  // name cannot be inferred from the id.
  std::map<std::string, std::unique_ptr<KitTimerEpochInfo>> epochInfo;

  // The actual times that have been recorded.
  std::vector<std::unique_ptr<KitTimerEpochImpl>> epochs;

public:
  using Iterator = PtrIterator<decltype(epochs)::const_iterator>;

public:
  const KitTimerEpochInfo &registerEpoch(const std::string &name) {
    std::lock_guard<std::mutex> guard(mtx);

    decltype(epochInfo)::const_iterator it = epochInfo.find(name);
    if (it != epochInfo.end())
      return *it->second;

    KitTimerID id = epochInfo.size() + 1;
    auto info = std::make_unique<KitTimerEpochInfo>(id, name);

    // This returns a pair of an iterator and a boolean. The iterator itself is
    // a pair consisting of the key and the value. We want the value here, so
    // we have `first->second` at the end. The value is a `std::unique_ptr`, but
    // we must return a reference to that object, hence the dereference at the
    // start. Perfectly obvious, isn't it?
    return *epochInfo.emplace(name, std::move(info)).first->second;
  }

  KitTimerEpochImpl *addEpoch(const KitTimerEpochInfo &info, KitThreadID thrd) {
    std::lock_guard<std::mutex> guard(mtx);

    // Emplace returns a reference to the unique pointer that was just added to
    // the epochs vector. We need to return the underlying pointer, so call
    // get() on the result.
    auto epoch = std::make_unique<KitTimerEpochImpl>(info, thrd);
    return epochs.emplace_back(std::move(epoch)).get();
  }

  bool empty() const { return epochs.empty(); }
  size_t size() const { return epochs.size(); }
  Iterator begin() const { return epochs.begin(); }
  Iterator end() const { return epochs.end(); }
};

} // namespace kitrt

using namespace kitrt;

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

static void writeTiming(FILE *fp, const KitTimerEpochImpl &epoch) {
  fprintf(fp, "\n      %ld", epoch.span());
}

static void writeEpochs(FILE *fp,
                        const std::vector<const KitTimerEpochImpl *> &epochs) {
  std::optional<KitTimerID> currEpoch = std::nullopt;
  std::optional<KitThreadID> currThrd = std::nullopt;
  bool firstThrd = false;

  fprintf(fp, "{");
  for (const KitTimerEpochImpl *epoch : epochs) {
    if (currEpoch != epoch->id()) {
      if (currEpoch) {
        fprintf(fp, "\n    ]");
        fprintf(fp, "\n  },");
      }
      fprintf(fp, "\n  \"%s\": {", epoch->name().c_str());
      currEpoch = epoch->id();
      currThrd = std::nullopt;
    }

    if (currThrd != epoch->thrd()) {
      if (currThrd)
        fprintf(fp, "\n    ],");
      fprintf(fp, "\n    \"%ld\": [", epoch->thrd());
      currThrd = epoch->thrd();
      firstThrd = true;
    }

    if (!firstThrd)
      fprintf(fp, ",");

    writeTiming(fp, *epoch);
    firstThrd = false;
  }
  fprintf(fp, "\n    ]");
  fprintf(fp, "\n  }");
  fprintf(fp, "\n}");
  fprintf(fp, "\n");
  fclose(fp);
}

static void writeTimings(const KitTimerContext &ctx) {
  if (ctx.empty())
    return;

  std::vector<const KitTimerEpochImpl *> epochs;
  for (const KitTimerEpochImpl &epoch : ctx)
    epochs.push_back(&epoch);

  std::stable_sort(
      epochs.begin(), epochs.end(),
      [](const KitTimerEpochImpl *l, const KitTimerEpochImpl *r) -> bool {
        if (l->id() < r->id())
          return true;
        else if (l->id() == r->id())
          return l->thrd() < r->thrd();
        return false;
      });

  if (FILE *fp = getFile())
    writeEpochs(fp, epochs);
}

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

  writeTimings(KitTimerContext::getSingleton());
  KitTimerContext::delSingleton();

  LOG("Finalized Kitsune timing context");
}
