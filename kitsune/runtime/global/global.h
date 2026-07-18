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

#include <cassert>
#include <cstdint>

namespace kitrt {

class KitCudaContext;
class KitHipContext;
class KitOCilkContext;
class KitOMPContext;
class KitPAPIContext;
class KitPthrContext;
class KitQthrContext;
class KitSerContext;
class KitTimerContext;

/// The context object for the global data used in kitrt. This will own the
/// globals used by all Kitsune's runtimes - even if they are only ever used in
/// a single runtime.
class KitRTContext {
public:
  /// Has this object been initialized.
  uint32_t initialized : 1;

  /// Has verbose mode been enabled.
  uint32_t verbose : 1;

  /// Global context for Kitsune's cuda runtime.
  KitCudaContext *cuda = nullptr;

  /// Global context for Kitsune's hip runtime.
  KitHipContext *hip = nullptr;

  /// Global context for Kitsune's OpenCilk runtime.
  KitOCilkContext *ocilk = nullptr;

  /// Global context for Kitsune's OpenMP runtime.
  KitOMPContext *omp = nullptr;

  /// Global context for Kitsune's PAPI support.
  KitPAPIContext *papi = nullptr;

  /// Global context for Kitsune's pthreads runtime.
  KitPthrContext *pthr = nullptr;

  /// Global context for Kitsune's Qthreads runtime.
  KitQthrContext *qthr = nullptr;

  /// Global context for Kitsune's serial runtime.
  KitSerContext *serial = nullptr;

  /// Global context for Kitsune's timer support.
  KitTimerContext *timer = nullptr;

private:
  template <typename T> T *take(T *&ptr) {
    T *ret = ptr;
    ptr = nullptr;
    return ret;
  }

  /// Get a singleton context object of the given type.
  template <typename T> inline T *getContextImpl() const;

public:
  KitRTContext();
  KitRTContext(const KitRTContext &) = delete;
  KitRTContext(KitRTContext &&) = delete;
  ~KitRTContext() = default;

  KitRTContext &operator=(const KitRTContext &) = delete;
  KitRTContext &&operator=(KitRTContext &&) = delete;

  /// Set the initialized flag on the object.
  void setInitialized(bool initialized);

  /// Set the verbose flag.
  void setVerbose(bool verbose);

  /// Set the global context object for Kitsune's cuda runtime.
  void addContext(KitCudaContext *ctx);

  /// Set the global context object for Kitsune's hip runtime.
  void addContext(KitHipContext *ctx);

  /// Set the global context object for Kitsune's OpenCilk runtime.
  void addContext(KitOCilkContext *ctx);

  /// Set the global context object for Kitusne's OpenMP runtime.
  void addContext(KitOMPContext *ctx);

  /// Set the global context object for Kitsune's PAPI support.
  void addContext(KitPAPIContext *ctx);

  /// Set the global context object for Kitsune's pthreads runtime.
  void addContext(KitPthrContext *ctx);

  /// Set the global context object for Kitsune's Qthreads runtime.
  void addContext(KitQthrContext *ctx);

  /// Set the global context object for Kitsune's serial runtime.
  void addContext(KitSerContext *ctx);

  /// Set the global context object for Kitsune's timer support.
  void addContext(KitTimerContext *ctx);

  /// Return the global context object of type \tparam T and set the
  /// corresponding field to nullptr.
  template <typename T> T *takeContext();

  /// Check if a context object of the given type exists.
  template <typename T> inline bool hasContext() const {
    return getContextImpl<T>();
  }

  /// Get the context object of the given type. The object must have been set.
  template <typename T> inline const T &getContext() const {
    assert(getContextImpl<T>() && "Global singleton context must be set");
    return *getContextImpl<T>();
  }

  /// Get the context object of the given type. The object must have been set.
  template <typename T> inline T &mutContext() {
    assert(getContextImpl<T>() && "Global singleton context must be set");
    return *getContextImpl<T>();
  }
};

template <> inline KitCudaContext *KitRTContext::getContextImpl() const {
  return cuda;
}

template <> inline KitHipContext *KitRTContext::getContextImpl() const {
  return hip;
}

template <> inline KitOCilkContext *KitRTContext::getContextImpl() const {
  return ocilk;
}

template <> inline KitOMPContext *KitRTContext::getContextImpl() const {
  return omp;
}

template <> inline KitPAPIContext *KitRTContext::getContextImpl() const {
  return papi;
}

template <> inline KitPthrContext *KitRTContext::getContextImpl() const {
  return pthr;
}

template <> inline KitQthrContext *KitRTContext::getContextImpl() const {
  return qthr;
}

template <> inline KitSerContext *KitRTContext::getContextImpl() const {
  return serial;
}

template <> inline KitTimerContext *KitRTContext::getContextImpl() const {
  return timer;
}

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
KitRTContext &mutKitRTContext();

} // namespace kitrt

#endif // KITRT_GLOBAL_GLOBAL_H
