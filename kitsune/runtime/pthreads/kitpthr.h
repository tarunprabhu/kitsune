//===- kitpthr.h - Runtime for Kitsune's pthreads tapir target --*- C++ -*-===//
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
// This targets the POSIX threads (pthreads) library.
//
// NOTE: This is an internal header and not really intended for public
// consumption. The main reason for this is to make the lit tests a bit more
// convenient to write.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_PTHREADS_KITPTHR_H
#define KITRT_PTHREADS_KITPTHR_H

#include "common/thread.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/// The function that is launched on every thread launched by pthreads.
typedef void (*KitPthrThrdFunc)(uint64_t start, uint64_t end, void *args);

/// An opaque launch context object. An instance is created by
/// \ref __kitpthr_async_launch, This should be passed to __kitpthr_sync where
/// it will be deleted.
struct KitPthrLaunchContext;

/// Launch some number of threads each of which will execute some number of
/// iterations in the space [\p start, \p end). This blocks until all threads
/// have completed. The compiler will transform all tapir loops so they are of
/// the following form:
///
///     unsigned numThreads = __kitpthr_num_threads();
///     size_t itersPerThread = (numThreads + n - 1) / numThreads
///     forall (unsigned t = 0; t < numThrds; ++t) {
///       size_t start = t * itersPerThread;
///       size_t end = std::min(start + itersPerThread, n);
///       for (size_t i = start; i < end; ++i)
///         ...
///     }
///
/// This function, therefore, will launch exactly `end - start - 1` threads,
/// each of which will execute exactly one iteration. The main thread will
/// execute the remaining iteration. It will, therefore, block until that
/// iteration has completed. In the future, `end - start` may be less than the
/// number of threads available.
///
/// \param f The function to execute on each thread
/// \param start The start index of the iteration space
/// \param end The value one greater than the last index of the iteration space
/// \param args A struct containing data to be passed to \p f
/// \params argSize The size of the underlying struct pointed to by \p args
/// \return An opaque thread context object. It is the caller's responsibility
/// to call \ref __kitpthr_sync with this context object. If no threads are
/// launched, i.e. \p f is run on the main thread, nullptr will be returned
/// instead. In this case, the caller is not required to call
/// \ref __kitpthr_sync.
KitPthrLaunchContext *__kitpthr_async_launch(KitPthrThrdFunc f, uint64_t start,
                                             uint64_t end, void *args,
                                             uint32_t argSize);

/// Get the number of threads available for parallel work.
uint64_t __kitpthr_num_threads(void);

/// Get the ID of the thread from which this is called.
KitThreadID __kitpthr_thread_id(void);

/// Join the threads launched by a previous call to \ref __kitpthr_async_launch.
/// \p ctx is the context returned by that call. \p ctx may be nullptr, in which
/// case, this function does nothing.
void __kitpthr_sync(KitPthrLaunchContext *ctx);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // KITRT_PTHREADS_KITPTHR_H
