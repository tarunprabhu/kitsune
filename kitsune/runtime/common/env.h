//===- env.h - Utilities to get and set environment variables --*- C++ -*--===//
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
// We don't bother with a C interface for these because they are only intended
// to be used within this runtime, and all of it is C++.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_COMMON_ENV_H
#define KITRT_COMMON_ENV_H

#include <optional>
#include <string>

namespace kitrt {

/**
 * Return `true` if either a variable named \p var, or, if it has been provided,
 * one named \p alt () has been set in the environment. Return `false`
 * otherwise.
 */
bool envContains(const std::string &var, const std::string &alt = "");

/**
 * Read the value of the variable \p var from the environment and return it
 * as-is. If the variable does not exist, and \p alt has been given, read the
 * value of \p alt from the environment. If that does not exist either, return
 * `std::nullopt`.
 */
std::optional<std::string> envLookup(const std::string &var,
                                     const std::string &alt = "");

/**
 * Read the value of the variable \p var from the environment. It is expected to
 * be of type \tparam T. If the variable does not exist in the environment, or
 * if it cannot be parsed into a value of type \tparam T, and \p alt has been
 * provided, perform \p alt. If that, too, fails to yield a valid value of type
 * \tparam T, return `std::nullopt`.
 */
template <typename T, std::enable_if_t<std::is_scalar_v<T>, int> = 0>
std::optional<T> envLookup(const std::string &var, const std::string &alt = "");

/**
 * Set a variable to the given value in the environment. If the variable has
 * already been set in the environment, the value will be overridden. Note that
 * if the value of the environment variable has already been read by some other
 * part of the runtime, that value will be unaffected.
 *
 * NOTE: This is only available on POSIX systems, but those are the only ones
 * that we currently support.
 */
void envSet(const std::string &var, const std::string &s);

/**
 * Set a variable to the given value in the environment. If the variable has
 * already been set in the environment, the value will be overridden. Note that
 * if the value of the environment variable has already been read by some other
 * part of the runtime, that value will be unaffected.
 *
 * NOTE: This is only available on POSIX systems, but those are the only ones
 * that we currently support.
 */
template <typename T, std::enable_if_t<std::is_scalar_v<T>, int> = 0>
void envSet(const std::string &var, const T &value);

/**
 * Unset the value of an environment variable.
 * NOTE: This is only available on POSIX systems, but those are the only ones
 * that we support currently.
 */
void envUnset(const std::string &var);

} // namespace kitrt

#endif // KITRT_COMMON_ENV_H
