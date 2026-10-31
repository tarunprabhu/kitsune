//===- utlis.h - Miscellaneous utilities ------------------------*- C++ -*-===//
//
// Copyright (c) 2021, Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021. Los Alamos National Security, LLC. This software was
//  produced under U.S. Government contract DE-AC52-06NA25396 for Los
//  Alamos National Laboratory (LANL), which is operated by Los Alamos
//  National Security, LLC for the U.S. Department of Energy. The
//  U.S. Government has rights to use, reproduce, and distribute this
//  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
//  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
//  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
//  derivative works, such modified software should be clearly marked,
//  so as not to confuse it with the version available from LANL.
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
// Some useful utilities that are used by several tapir-target-specific
// runtimes.
//
// TODO: Rather than just dumping a number of unrelated utilities here, we
// should try to find a more reasonable organization across source files.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_COMMON_UTILS_H
#define KITRT_COMMON_UTILS_H

#include <stdint.h>
#include <utility>

#include "kitsune/Shared/RTNestUtils.h"

namespace kitrt {

/// A loop nest context object. This is created just before beginning execution
/// of a tapir loop nest. Currently, this is only used when executing a nest on
/// a CPU, but it may also be used for a GPU. Given the number of threads
/// available on the system for parallel execution, this divides the number of
/// threads among the loops in the nest. This object is then used by each thread
/// to compute the actual bounds of each loop in the nest.
///
/// This object is intended to be opaque to any callers of the runtime.
struct LoopNestContext {
public:
  /// The depth of the nest.
  const uint32_t depth;

  /// The number of threads for each loop in the nest. We currently only support
  /// tapir loop nests of depth <= 3. Only the array elements at index < `depth`
  /// are valid. The values of the remaining elements are undefined, though they
  /// are likely to be set to 1.
  const uint32_t threads[NEST_MAX_DEPTH];

  /// The number of threads that are used to execute the loop nest described by
  /// this context.
  const uint64_t usedThreads;

public:
  LoopNestContext(uint32_t depth, uint32_t t0, uint32_t t1 = 1, uint32_t t2 = 1)
      : depth(depth), threads{t0, t1, t2}, usedThreads(t0 * t1 * t2) {}

  uint32_t operator[](uint32_t i) const { return threads[i]; }
};

using LoopBounds = std::pair<uint64_t, uint64_t>;

/// Get the number of parallel execution threads to use. This is determined as
/// follows:
///
///   - If KIT_NUM_THREADS was set in the environment to a valid value, return
///     that.
///
///   - Otherwise, if \p alternate is not nullptr, and it is set in the
///     environment to a valid value, return that.
///
///   - Otherwise, return the value obtained by calling \ref getNumCPUs. This
///     is guaranteed to be at least 1.
///
/// A value is valid if it is a positive, base 10 integer, whose value is at
/// most 2^31 - 1.
uint32_t getNumThreadsOrCPUs(const char *alternate = nullptr);

/// Get the number of CPU cores on the system. If this cannot be determined for
/// any reason, return 1.
uint32_t getNumCPUs(void);

/// Get the nearest power of 2 that is less than or equal to \p n.
uint32_t nearestPowerOf2LE(uint32_t n);

/// Get the nearest multiple of \p m that is less than or equal to \p n. If
/// either \p m, or \p n is 0, or if \p m is greater than \p n, return \p m.
uint32_t nearestMultipleLT(uint32_t m, uint32_t n);

/// Write a stack trace to stderr.
void printStackTrace(void);

/// Create a loop nest context object for a nest of depth \p depth.
/// \p totalThreads is the number of threads available to execute the nest.
LoopNestContext *nestBegin(uint64_t totalThreads, uint32_t depth);

/// Delete a loop nest context object previously created by \ref nestBegin.
void nestEnd(LoopNestContext *ctx);

/// Get the number of threads that will be used for parallel execution of a loop
/// nest described by the given context. The number of threads used may be
/// smaller than the number available on the system.
uint64_t nestThreads(const LoopNestContext &ctx);

/// Get the bounds of the loop at level \p level in a nest described by the
/// context \p ctx that will be executed by the thread with index \p tid. \p tid
/// must be in `[0, nestThreads(ctx))`. \p beg and \p end are the original
/// bounds of the loop.
LoopBounds nestBounds(const LoopNestContext &ctx, uint64_t tid, uint64_t beg,
                      uint64_t end, uint32_t level);

} // namespace kitrt

#endif // KITRT_COMMON_UTILS_H
