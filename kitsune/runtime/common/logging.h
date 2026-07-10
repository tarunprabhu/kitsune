//===- logging.h - Logging utilites for Kitsune's runtime -------*- C++ -*-===//
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
// For now, these just write to stderr. We probably don't need anything more
// sophisticated than this.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_COMMON_LOGGING_H
#define KITRT_COMMON_LOGGING_H

namespace kitrt {

/**
 * Print an error message to stderr and terminate the process with an exit code.
 * \p msg may be a printf-compatible format string. In that case, any optional
 * arguments must be of the appropriate types.
 */
[[noreturn]] void fatal(const char *label, const char *msg, ...);

/**
 * Print an error message to stderr. \p msg may be a printf-compatible format
 * string. In that case, any optional arguments must be of the appropriate
 * types.
 */
void error(const char *label, const char *msg, ...);

/**
 * Print a warning message to stderr. \p msg may be a printf-compatible format
 * string. In that case, any optional arguments must be of the appropriate
 * types.
 */
void warn(const char *label, const char *msg, ...);

/**
 * Print a message to stderr if verbose mode has been enabled. \p msg may be a
 * printf-compatible format string. In that case, any optional arguments must be
 * of the appropriate types.
 */
void log(const char *label, const char *msg, ...);

/**
 * Print an error message to stderr. \p msg may be a printf-compatible format
 * string. In that case, any optional arguments must be of the appropriate
 * types. This does not add a trailing newline after printing the message.
 */
void errorNoEndl(const char *label, const char *msg, ...);

/**
 * Print a warning message to stderr. \p msg may be a printf-compatible format
 * string. In that case, any optional arguments must be of the appropriate
 * types. This does not add a trailing newline after printing the message.
 */
void warnNoEndl(const char *label, const char *msg, ...);

/**
 * Print a message to stderr if verbose mode has been enabled. \p msg may be a
 * printf-compatible format string. In that case, any optional arguments must be
 * of the appropriate types. This does not add a trailing newline after printing
 * the message.
 */
void logNoEndl(const char *label, const char *msg, ...);

} // namespace kitrt

#endif // KITRT_COMMON_LOGGING_H
