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

#include <cassert>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

/**
 * Initialize the core kitsune runtime components that are shared
 * across all the target runtimes.  Note that this call is typically
 * invoked by each specific runtime target (e.g., CUDA) vs. having
 * this call initialization each target runtime.  It can be called
 * multiple times as it is guarded to avoid repeated initialization.
 */
void __kitrt_initialize();

/**
 * Finalize Kitsune's runtime. This is typically called from the global
 * destructors for individual runtimes such as kitcuda, or kitomp. This can be
 * safely called multiple times.
 */
void __kitrt_finalize();

/**
 * Set the runtime system to operate in verbose mode.
 */
void __kitrt_enable_verbose_mode();

/**
 * Disable the runtime system's verbose reporting mode.
 */
void __kitrt_disable_verbose_mode();

/**
 * Enable/disable the runtime's verbose mode. __kitrt_initialize() reads the
 * value of the KITRT_VERBOSE environment variable. This should only be used
 * if there is need to set that value after the runtime has been initialized.
 *
 * @param enable - if `true` enable verbose mode, disable if `false`.
 */
void __kitrt_set_verbose_mode(bool enable);

/**
 * Check if the verbose mode has been enabled in Kitsune's runtime.
 */
inline bool __kitrt_verbose_mode() {
  extern bool _kitrt_verbose_mode;
  return _kitrt_verbose_mode;
}

/**
 * Provide a backtrace to stderr to help track down runtime crashes.
 */
void __kitrt_print_stack_trace();

/**
 * Set a variable to the given value in the environment. If the variable has
 * already been set in the environment, the value will be overridden. If any
 * part of the runtime has read the old value, that value will not be changed.
 */
void __kitrt_set_env(const char *varname, const char *value);

/**
 * Unset the value of an environment variable.
 * NOTE: This is only available on POSIX systems, but those are the only ones
 * that we support currently.
 */
void __kitrt_unset_env(const char *varname);

/**
 * Print an error message to stderr and terminate the process with an exit code.
 * \p msg may be a printf-compatible format string. In that case, any optional
 * arguments must be of the appropriate types.
 */
[[noreturn]] void __kitrt_fatal(const char *label, const char *msg, ...);

/**
 * Print an error message to stderr. \p msg may be a printf-compatible format
 * string. In that case, any optional arguments must be of the appropriate
 * types.
 */
void __kitrt_error(const char *label, const char *msg, ...);

/**
 * Print a warning message to stderr. \p msg may be a printf-compatible format
 * string. In that case, any optional arguments must be of the appropriate
 * types.
 */
void __kitrt_warn(const char *label, const char *msg, ...);

/**
 * Print an error message to stderr if verbose mode has been enabled. \p msg may
 * be a printf-compatible format string. In that case, any optional arguments
 * must be of the appropriate types.
 */
void __kitrt_message(const char *label, const char *msg, ...);

/**
 * Print an error message to stderr if verbose mode has been enabled. \p msg may
 * be a printf-compatible format string. In that case, any optional arguments
 * must be of the appropriate types. This doe not add a trailing newline after
 * printing the message.
 */
void __kitrt_message_noflush(const char *label, const char *msg, ...);

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
 * Get the number of threads to use for parallel work from the environment
 * variable named KIT_NUM_THREADS. Returns 0 if any of the following is true:
 *
 *  - `KIT_NUM_THREADS` is not set in the environment
 *  - `KIT_NUM_THREADS` is set, but the value is not a positive, decimal integer
 *  - The value of `KIT_NUM_THREADS` is a positive, decimal integer, but its
 *    value is greater than 2^31 - 1
 *
 * Otherwise, returns the value of the environment variable parsed into a 32-bit
 * unsigned integer.
 */
unsigned __kitrt_num_threads_from_env();

/**
 * Get the number of CPU cores on the system. If the number could not be
 * determined for any reason, returns 1.
 */
unsigned __kitrt_num_cpus();

#ifdef __cplusplus
} // extern "C"
#endif

/**
 * The environment variable that can be used to control the degree of CPU
 * parallelism. This must be set to a positive, decimal integer that specifies
 * the number of threads/workers to use. In most cases, this environment
 * variable will be queried in a global constructor and an appropriate mechanism
 * will be used to control the behavior of the underlying runtime.
 */
static constexpr const char *__kitrt_envname_num_threads = "KIT_NUM_THREADS";

/**
 * Read the value of an environment variable. If the variable does not exist in
 * the environment return `false`. Otherwise, return `true` and populate
 * \p value with the parsed value of the environment variable.
 */
template <typename ValueType>
bool __kitrt_get_env_value(const char *varname, ValueType &value);

#endif // __KITRT_H__
