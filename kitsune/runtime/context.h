//===- context.h - Context object for global state -------------*- C++ -*--===//
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
// Context object for global state used in the runtime.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_CONTEXT_H
#define KITRT_CONTEXT_H

#include "kitsune/Shared/RTID.h"
#include "runtimes.h"

#include <cassert>
#include <cstdint>
#include <map>

namespace kitrt {

/// The context object for the global state used in kitrt. Individual runtimes
/// may also use context objects containing global state. This will own those
/// context objects as well.
class Context {
private:
  /// Has this object been initialized.
  uint32_t initialized_ : 1;

  /// Has verbose mode been enabled.
  uint32_t verbose_ : 1;

  /// Enable colorized output. This is only used for logs and diagnostic
  /// messages.
  uint32_t colors_ : 1;

  /// Each time, initialization of a runtime is requested, the corresponding
  /// entry in this map is incremented. When finalization is requested, the
  /// entry is decremented. Only if, after decrementing, the value becomes zero,
  /// is the runtime actually finalized.
  std::map<RTID, unsigned> refCounts;

  /// The runtime context objects.
  ContextsTuple ctxs;

public:
  Context() = default;
  Context(const Context &) = delete;
  Context(Context &&) = delete;
  ~Context() = default;

  Context &operator=(const Context &) = delete;
  Context &operator=(Context &&) = delete;

  /// Set the initialized flag on the object.
  inline void setInitialized(bool initialized) { initialized_ = initialized; }

  /// Set the verbose flag.
  inline void setVerbose(bool verbose) { verbose_ = verbose; }

  /// Set the colors flag.
  inline void setColors(bool colors) { colors_ = colors; }

  inline bool initialized() const { return initialized_; }
  inline bool verbose() const { return verbose_; }
  inline bool colors() const { return colors_; }

  /// Increment the refcount for the given runtime. Returns the updated
  /// refCount. If an entry for the RTID does not exist, it will be added and
  /// set to 1.
  unsigned incr(RTID rt) { return refCounts[rt] += 1; }

  /// Decrement the refcount for the given runtime. Returns the updated
  /// refcount. An entry for the refcount is assumed to be present.
  unsigned decr(RTID rt) { return refCounts.at(rt) -= 1; }

  /// Ensure that the context object of the given type has been created.
  template <typename T> inline void ensure() const {
    assert(std::get<T *>(ctxs) && "Kitsune runtime initialized");
  }

  /// Check if a context object of the given type exists.
  template <typename T> inline bool initialized() const {
    return std::get<T *>(ctxs);
  }

  /// Add a global context object of type \tparam T.
  template <typename T> void add(T *ctx) {
    assert(!std::get<T *>(ctxs) &&
           "Kitsune runtime context object not registered");
    std::get<T *>(ctxs) = ctx;
  }

  /// Return the global context object of type \tparam T and set the
  /// corresponding field to nullptr.
  template <typename T> T *take() {
    assert(std::get<T *>(ctxs) && "Kitsune runtime context object registered");

    T *ctx = std::get<T *>(ctxs);
    std::get<T *>(ctxs) = nullptr;
    return ctx;
  }

  /// Get the context object of the given type. The object must have been set.
  template <typename T> inline const T &get() const {
    ensure<T>();
    return *std::get<T *>(ctxs);
  }

  /// Get the context object of the given type. The object must have been set.
  template <typename T> inline T &get() {
    ensure<T>();
    return *std::get<T *>(ctxs);
  }
};

template <> inline bool Context::initialized<Context>() const {
  return initialized();
}

// Specialize some traits here because there is no better place to do this.
// this.
template <> inline constexpr const char *rtname_v<RT_COMMON> = "common";
template <> struct context_t<RT_COMMON> {
  using type = Context;
};
template <> struct rtid_v<Context> : detail::rtid_v<RT_COMMON> {};

/// The singleton global context that contains the global data used by kitrt.
/// It is not clear what the benefits of exposing a constant reference are in
/// this case. It adds an unnecessary extra load when looking up anything in the
/// object.
extern Context gctx;

} // namespace kitrt

#endif // KITRT_CONTEXT_H
