//===- global.h - Singleton object for global data -------------*- C++ -*--===//
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
// Singleton object for global data.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_GLOBAL_GLOBAL_H
#define KITRT_GLOBAL_GLOBAL_H

#include "runtimes.h"

#include <cassert>
#include <cstdint>

namespace kitrt {

/// The context object for the global data used in kitrt. This will own the
/// globals used by all Kitsune's runtimes - even if they are only ever used in
/// a single runtime.
class KitRTContext {
public:
  /// Has this object been initialized.
  uint32_t initialized : 1;

  /// Has verbose mode been enabled.
  uint32_t verbose : 1;

  /// Enable colorized output. This is only used for logs and diagnostic
  /// messages.
  uint32_t colors : 1;

  /// The runtime context objects.
  ContextsTuple ctxs;

public:
  KitRTContext() = default;
  KitRTContext(const KitRTContext &) = delete;
  KitRTContext(KitRTContext &&) = delete;
  ~KitRTContext() = default;

  KitRTContext &operator=(const KitRTContext &) = delete;
  KitRTContext &operator=(KitRTContext &&) = delete;

  /// Set the initialized flag on the object.
  void setInitialized(bool initialized);

  /// Set the verbose flag.
  void setVerbose(bool verbose);

  /// Set the colors flag.
  void setColors(bool colors);

  /// Ensure that the context object of the given type has been created.
  template <typename T> inline void ensure() const {
    assert(std::get<T *>(ctxs) && "Kitsune runtime initialized");
  }

  /// Check if a context object of the given type exists.
  template <typename T> inline bool has() const { return std::get<T *>(ctxs); }

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

/// The singleton global context that contains the global data used by kitrt.
/// We expose the structure of the object to make accesses to some global data
/// fast. Because const in C++ does not imply transitively const, this only
/// means that the scalar fields of the objects and the pointers themselves may
/// not be changed. However, the pointees of objects may still change, so it is
/// possible to call non-const methods on those objects.
extern const KitRTContext &gctx;

/// Get a non-const reference to the single global context. This should
/// generally only be used in the various *_initialize functions that are called
/// from global constructors. This function is not thread-safe.
KitRTContext &mutCtx();

/// Get a constant reference to a runtime context object. This cannot be used to
/// get a reference to the singleton object. The requested object must have been
/// initialized before this is called.
template <typename T> inline const T &getCtx() { return gctx.get<T>(); }

/// Get a non-const reference to a runtime context object. This cannot be used
/// to get a reference to the singleton global. The requested object must have
/// been initialized before this is called.
template <typename T> inline T &mutCtx() { return mutCtx().get<T>(); }

} // namespace kitrt

#endif // KITRT_GLOBAL_GLOBAL_H
