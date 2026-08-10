//==- instr.h - Utilities for Kitsune's instrumentation support --*- C++ -*-==//
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
// Utilities for Kitsune's compiler-inserted instrumentation that is backed by
// the runtime.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_COMMON_INSTR_H
#define KITRT_COMMON_INSTR_H

#include "common/thread.h"

#include <cassert>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace kitrt {

/// Base class for Epoch objects. This simply provides some wrappers around the
/// members of the base class info objects.
class EpochBase {
protected:
  const char *const name_;
  const KitThreadID thrd_;

protected:
  EpochBase(const char *name, KitThreadID thrd) : name_(name), thrd_(thrd) {}

public:
  const char *name() const { return name_; }
  KitThreadID thrd() const { return thrd_; }
};

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
/// long as the thread ID in each case is different.
///
/// The same callsite may be visited multiple times during program execution.
/// The most basic case is when a callsite is in a loop. In this case, the
/// events recorded during that epoch will be merged with the events recorded in
/// previous earlier visits to the epoch. In order to record events per-visit,
/// set the KIT_INSTR_SEPARATE environment variable. The value of this variable
/// is not relevant, only its presence is necessary.
///
/// WARNING: The name of an epoch *MUST* be a string-literal. Unfortunately,
/// there is no way for the runtime to enforce this. Failure to use a string
/// literal may in incorrect recording of events, or a catastrophic runtime
/// failure. Kitsune will ensure that string literals are used when
/// automatically inserting instrumentation.
///
template <typename T, typename EpochT> class InstrBase {
protected:
  using Epoch = EpochT;
  using EpochID = std::pair<const char *, KitThreadID>;

private:
  // Each element of the vector is a unique_ptr so references to the epoch
  // remain valid even if the vector is resized.
  using OwnedEpochs = std::vector<std::unique_ptr<Epoch>>;

protected:
  // If this is true, the events measured by each epoch will be recorded
  // separately. Otherwise, the events will be accumulated. The default is to
  // accumulate events in each epoch.
  bool separate = false;

  // A mutex that controls all accesses to the mutable members of this class.
  std::mutex mtx;

  // If the events that occur during multiple visits to a callsite are
  // accumulated, a single epoch will be present for each EpochID. Otherwise, an
  // epoch will be created for each visit to a callsite.
  std::map<EpochID, OwnedEpochs> epochs;

protected:
  InstrBase();

public:
  template <typename... Args>
  inline Epoch *addEpoch(const char *name, KitThreadID thrd, Args &&...args) {
    assert(name && "Name of an epoch must not be NULL");

    std::lock_guard<std::mutex> guard(mtx);

    // try_emplace will have no effect if the epoch already exists, otherwise,
    // it will add a new entry with the epoch ID and an empty vector.
    EpochID id = {name, thrd};
    epochs.try_emplace(id);

    // If we are recording each visit of an epoch separately, then we need to
    // create a new object each time. Otherwise, we only need to create it the
    // first time this epoch is visited.
    OwnedEpochs &vec = epochs.at(id);
    if (separate || vec.empty())
      vec.emplace_back(static_cast<T *>(this)->makeEpoch(name, thrd, args...));
    return vec.back().get();
  }

  void writeJSON(const char *outFileEnvVar) const;
};

} // namespace kitrt

#endif // KITRT_COMMON_INSTR_H
