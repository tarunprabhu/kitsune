//===- kittimer.h - Utilities to collect timings ---------------*- C++ -*--===//
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

#ifndef KITRT_KITTIMER_H
#define KITRT_KITTIMER_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/// Unique identifier for a timer.
using TimerID = uint64_t;

/// The ID of the thread in which a timer is running.
using ThreadID = uint64_t;

/// Initialize the Kitsune's timing context.
void __kittimer_initialize(void);

/// Cleanup Kitsune's timing context. If any timings were collected, print them
/// to stdout.
void __kittimer_finalize(void);

/// Start the timer \p timer. \p thrd is the ID of the thread on which the
/// timer is running. \p name is the name of the timer. The runtime will create
/// a mapping between \p timer and \p name, but only if \p timer was not used
/// in an earlier call to this function. Multiple threads can share the a timer
/// ID. This is useful when measuring the times for individual threads launched,
/// for instance, when a parallel loop is lowered using a CPU-centric tapir
/// target.
void __kittimer_tick(TimerID timer, ThreadID thrd, const char *name);

/// Stop the timer \p timer running on a thread with ID \p thrd.
void __kittimer_tock(TimerID timre, ThreadID thrd);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // KITRT_KITTIMER_H
