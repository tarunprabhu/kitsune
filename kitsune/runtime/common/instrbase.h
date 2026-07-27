//===- instrbase.h - Base class for runtime instrumentation -----*- C++ -*-===//
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
// A base class for compiler-inserted instrumentation backed by Kitsune's
// runtime.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_COMMON_INSTR_BASE_H
#define KITRT_COMMON_INSTR_BASE_H

#include "common/instrutils.h"
#include "global/singleton.h"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace kitrt {

/// Base class for compiler-inserted instrumentation backed by Kitsune's
/// runtime. This provides the common infrastructure to create epochs and write
/// the recorded information to JSON. This is agnostic of the actual
/// instrumentation that is recorded. That is handled entirely by the derived
/// classes that inherit from this.
///
/// Kitsune's instrumentation support is centered around epochs. An epoch is a
/// Single-Entry-Single-Exit region of code that is bounded by a pair of "start"
/// and "stop" functions. Some external quantities may be recorded when the
/// code bounded by these functions is executed - for instance, wall-clock time
/// or hardware counters. An epoch is created and return by the "start" function
/// and must be passed to the corresponding "stop" function.
///
/// Each epoch is assigned a user-provided name. This is nearly always passed as
/// an argument to the "start" function. This name ought to be something that is
/// meaningful to the user, but it need not be. However, the names *MUST* be
/// globally unique, i.e. during the execution of a program, there may not be
/// more than one callsite that calls "start" with the same name. Internally, an
/// epoch is uniquely identified by a pair of the name and the ID of the thread
/// in which the epoch was created. This implies that, in a multi-threaded
/// program, "start" can be invoked on multiple threads with the same name, as
/// long as the thread ID can be obtained when the epoch object is created.
///
/// By definition, this implies that an epoch is associated with a callsite.
/// \tparam EpochInfoT is an object that maintains metadata related to the
/// callsite from which an epoch is created. This will always have a name, but
/// may contain additional information.
///
/// The same callsite may be reached multiple times during program execution.
/// The most basic case is when a callsite is in a loop. In this case, a new
/// epoch object will be created each time the callsite is reached. In the final
/// output that is written out on program termination, any data collected during
/// these epochs will *NOT* be merged, but will be recorded separately.
template <typename T, typename EpochT, typename EpochInfoT, typename ThreadIDT>
class KitInstrBase : public KitContextMixin<T> {
protected:
  using Epoch = EpochT;
  using EpochInfo = EpochInfoT;
  using ThreadID = ThreadIDT;

protected:
  // A mutex that controls all accesses to the mutable members of this class.
  std::mutex mtx;

  // Information about each unique epoch.
  std::map<std::string, std::unique_ptr<const EpochInfo>> epochInfo;

  // All Epoch objects that have been created. If this were a vector of Epoch
  // objects, returning a reference to the element would be unsafe since, if the
  // vector were to be resized and and reallocated, that reference would no
  // longer be valid. By wrapping everything in a unique pointer, the vector may
  // be safely resized without invalidating any pointers or references.
  std::vector<std::unique_ptr<Epoch>> epochs;

private:
  void writeJSONHeader(FILE *fp) const { fprintf(fp, "{"); }

  void writeEpochHeader(FILE *fp, const Epoch &epoch) const {
    fprintf(fp, "\n  \"%s\": {", epoch.name().c_str());
  }

  void writeThreadHeader(FILE *fp, const Epoch &epoch) const {
    fprintf(fp, "\n    \"%ld\": [", epoch.thrd());
  }

  void writeEpoch(FILE *fp, const Epoch &epoch) const {
    T::writeEpoch(fp, epoch);
  }

  void writeThreadFooter(FILE *fp, const Epoch &epoch, bool comma) const {
    fprintf(fp, "\n    ]");
    if (comma)
      fprintf(fp, ",");
  }

  void writeEpochFooter(FILE *fp, const Epoch &epoch, bool comma) const {
    fprintf(fp, "\n  }");
    if (comma)
      fprintf(fp, ",");
  }

  void writeEpochs(FILE *fp, const std::vector<const Epoch *> &epochs) const {
    assert(!epochs.empty() && "Epochs must not be empty");

    writeEpochHeader(fp, *epochs[0]);
    writeThreadHeader(fp, *epochs[0]);
    writeEpoch(fp, *epochs[0]);
    for (unsigned i = 1, e = epochs.size(); i != e; ++i) {
      const Epoch &prev = *epochs[i - 1];
      const Epoch &curr = *epochs[i];
      if (prev.name() != curr.name()) {
        writeThreadFooter(fp, prev, /*comma=*/false);
        writeEpochFooter(fp, prev, /*comma=*/true);
        writeEpochHeader(fp, curr);
        writeThreadHeader(fp, curr);
      } else if (prev.thrd() != curr.thrd()) {
        writeThreadFooter(fp, prev, /*comma=*/true);
        writeThreadHeader(fp, curr);
      } else {
        // This is a different iteration on the same thread.
        fprintf(fp, ",");
      }
      writeEpoch(fp, curr);
    }
    writeThreadFooter(fp, *epochs[0], /*comma=*/false);
    writeEpochFooter(fp, *epochs[0], /*comma=*/false);
  }

  void writeJSONFooter(FILE *fp) const { fprintf(fp, "\n}"); }

  void writeJSON(FILE *fp, const std::vector<const Epoch *> &epochs) const {
    writeJSONHeader(fp);
    writeEpochs(fp, epochs);
    writeJSONFooter(fp);
    fprintf(fp, "\n");
  }

  void sortEpochs(std::vector<const Epoch *> &epochs) const {
    auto sortByNameThenThreadID = [](const Epoch *l, const Epoch *r) -> bool {
      if (l->name() < r->name())
        return true;
      else if (l->name() == r->name())
        return l->thrd() < r->thrd();
      return false;
    };

    // Sort the epochs by name, and then by thread id.
    // TODO: What we actually want here is to first group by name in the order
    // in which the epochs were added, then sort by thread id *only*. This will
    // ensure that the epochs appear in the output in the order in which they
    // were created during execution. This is, arguably, more useful than
    // sorting lexicographically by name. But the complexity of doing that is
    // probably not worth it since the JSON output is not really intended for
    // human consumption anyway.
    std::stable_sort(epochs.begin(), epochs.end(), sortByNameThenThreadID);
  }

public:
  template <typename... Args>
  const EpochInfo &registerEpoch(const std::string &name, Args &&...args) {
    std::lock_guard<std::mutex> guard(mtx);

    typename decltype(epochInfo)::const_iterator it = epochInfo.find(name);
    if (it != epochInfo.end())
      return *it->second;

    // The call to emplace returns a pair of an iterator and a boolean. The
    // iterator itself is a pair consisting of the key and the value. We want
    // the value here, so we have `first->second` at the end. The value is a
    // `std::unique_ptr`, but we need to return a reference to that object,
    // hence the dereference at the start. Perfectly obvious, isn't it?
    std::unique_ptr<EpochInfo> info(T::makeEpochInfo(name, args...));
    return *epochInfo.emplace(name, std::move(info)).first->second;
  }

  template <typename... Args>
  Epoch *addEpoch(const EpochInfo &info, Args &&...args) {
    std::lock_guard<std::mutex> guard(mtx);

    // The call to emplace returns a reference to the unique pointer that was
    // just added to the epochs vector. We want to return the underlying
    // pointer, so call get on the result.
    std::unique_ptr<Epoch> epoch(T::makeEpoch(info, args...));
    return epochs.emplace_back(std::move(epoch)).get();
  }

  void writeJSON(const char *outFileEnvVar) const {
    if (!epochs.empty()) {
      std::vector<const Epoch *> epochPtrs;
      for (const std::unique_ptr<Epoch> &epoch : epochs)
        epochPtrs.push_back(epoch.get());

      sortEpochs(epochPtrs);
      if (FILE *fp = getInstrumentationOutputFile(outFileEnvVar)) {
        writeJSON(fp, epochPtrs);
        if (fp != stdout && fp != stderr)
          fclose(fp);
      }
    }
  }
};

} // namespace kitrt

#endif // KITRT_COMMON_INSTR_BASE_H
