//=- kitrt.h - Routines common to several of Kitsune's runtimes --*- C++ -*--=//
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

#ifndef __KITRT_H__
#define __KITRT_H__

#include "common/kitpapi.h"
#include "common/timer.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

/**
 * Initialize the core kitsune runtime components that are shared across all the
 * tapir-target-specific runtimes. This is typically called in the global
 * constructor for each target-specific runtime. It is safe to call this
 * multiple times, thought this should be avoided.
 */
void __kitrt_initialize(void);

/**
 * Finalize Kitsune's runtime. This is typically called from the global
 * destructors for individual runtimes such as kitcuda, or kitomp. This can be
 * safely called multiple times.
 */
void __kitrt_finalize(void);

/**
 * Enable verbose mode. This will cause logging messages to be written to
 * standard error.
 */
void __kitrt_enable_verbose_mode(void);

/**
 * Disable verbose mode.
 */
void __kitrt_disable_verbose_mode(void);

/**
 * Enable/disable the runtime's verbose mode. __kitrt_initialize() reads the
 * value of the KIT_VERBOSE environment variable. This should only be used
 * if there is need to set that value after the runtime has been initialized.
 *
 * @param enable - if `true` enable verbose mode, disable if `false`.
 */
void __kitrt_set_verbose_mode(bool enable);

/**
 * Check if the verbose mode has been enabled in Kitsune's runtime.
 */
inline bool __kitrt_verbose_mode(void) {
  extern bool _kitrt_verbose_mode;
  return _kitrt_verbose_mode;
}

/**
 * Provide a backtrace to stderr to help track down runtime crashes.
 */
void __kitrt_print_stack_trace(void);

/**
 * Get the nearest power of 2 that is less than or equal to \p n.
 */
unsigned nearestPowerOf2LE(unsigned n);

/**
 * *** EXPERIMENTAL: This is a new interface between the compiler and
 * the runtime.  It is a quick set of details regarding the particular
 * instruction mix of a kernel and any device-side functions it calls.
 * It is gathered from the LLVM form of the code (not ptx/s-code) and
 * at this point is limited.  In general we are using to explore
 * impacts on launch parameters.
 * NOTE: Changing this structure has implications on code generation
 * inside the CudaABI component of the compiler -- both must be kept
 * up-to-date.
 */
typedef struct _kitrt_inst_mix_info {
  uint64_t numMemoryOps; // Number of memory (read/write) ops.
  uint64_t numFlops;     // Floating point operations.
  uint64_t numIntOps;    // Integer operations.
  uint64_t numOtherOps;  // Other operations.
} KitRTInstMix;

/**
 * Get the number of parallel execution threads to use. This is determined as
 * follows:
 *
 *   - If KIT_NUM_THREADS was set in the environment to a valid value, return
 *     that.
 *
 *   - Otherwise, if \p alternate is not NULL, and it is set in the environment
 *     to a valid value, return that.
 *
 *   - Otherwise, return the value obtained by calling `__kitrt_num_cpus`. This
 *     is guaranteed to be at least 1.
 *
 * A value is valid if it is a positive, base 10 integer, whose value is at most
 * 2^31 - 1.
 */
unsigned __kitrt_num_threads(const char *alternate);

/**
 * Get the number of CPU cores on the system. If this could not be determined,
 * return 1.
 */
unsigned __kitrt_num_cpus(void);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // __KITRT_H__
