//===- env.cpp - Utilities to get and set environment variables -----------===//
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

#include "common/env.h"
#include "common/logging.h"

#include <algorithm>
#include <cstdint>
#include <ctype.h>
#include <limits>
#include <stdexcept>

#define LABEL "kitrt"

using namespace kitrt;

bool kitrt::envContains(const std::string &var) { return getenv(var.c_str()); }

template <typename V, typename F, typename... Args>
static std::optional<V> parseAs(const std::string &s, const std::string &var,
                                F converter, Args &&...args) {
  try {
    std::size_t pos = 0;
    V tmp = converter(s, &pos, args...);
    if (pos == s.size())
      return tmp;
    warn(LABEL,
         "Ignoring environment variable '%s'. Value contains unparsed "
         "characters",
         var.c_str());
  } catch (std::invalid_argument) {
    warn(LABEL, "Ignoring environment variable '%s'. Value not valid",
         var.c_str());
  } catch (std::out_of_range) {
    warn(LABEL, "Ignoring environment variable '%s'. Value not in range",
         var.c_str());
  }
  return std::nullopt;
}

template <typename V>
static std::optional<V> parseAs(const std::string &str, const std::string &var);

template <>
std::optional<bool> parseAs(const std::string &s, const std::string &var) {
  auto equals = [](const std::string &l, const std::string &r) -> bool {
    return std::equal(l.begin(), l.end(), r.begin(),
                      [](unsigned char cl, unsigned char cr) -> bool {
                        return std::tolower(cl) == std::tolower(cr);
                      });
  };

  if (equals(s, "true") || s == "1")
    return true;
  else if (equals(s, "false") || s == "0")
    return false;

  // FIXME: We should be more strict and reject this, but for now, we are
  // permissive.
  warn(LABEL,
       "Environment variable '%s' not set to known boolean. Assuming 'true'",
       var.c_str());
  return true;
}

template <>
std::optional<int32_t> parseAs(const std::string &s, const std::string &var) {
  using Converter = int32_t(const std::string &, std::size_t *, int);
  return parseAs<int32_t>(s, var, (Converter *)&std::stoi, /*base=*/10);
}

template <>
std::optional<uint32_t> parseAs(const std::string &s, const std::string &var) {
  using Converter = uint64_t(const std::string &, std::size_t *, int);
  if (std::optional<uint64_t> tmp =
          parseAs<uint64_t>(s, var, (Converter *)&std::stoul, /*base=*/10))
    if (tmp <= std::numeric_limits<uint32_t>::max())
      return tmp;
  return std::nullopt;
}

template <>
std::optional<int64_t> parseAs(const std::string &s, const std::string &var) {
  using Converter = int64_t(const std::string &, std::size_t *, int);
  return parseAs<int64_t>(s, var, (Converter *)&std::stol, /*base=*/10);
}

template <>
std::optional<uint64_t> parseAs(const std::string &s, const std::string &var) {
  using Converter = uint64_t(const std::string &, std::size_t *, int);
  return parseAs<uint64_t>(s, var, (Converter *)&std::stoul, /*base=*/10);
}

template <>
std::optional<float> parseAs(const std::string &s, const std::string &var) {
  using Converter = float(const std::string &, std::size_t *);
  return parseAs<float>(s, var, (Converter *)&std::stof);
}

template <>
std::optional<double> parseAs(const std::string &s, const std::string &var) {
  using Converter = double(const std::string &, std::size_t *);
  return parseAs<double>(s, var, (Converter *)&std::stod);
}

std::optional<std::string> kitrt::envLookup(const std::string &var) {
  if (char *s = getenv(var.c_str()))
    return s;
  return std::nullopt;
}

template <typename T, std::enable_if_t<std::is_scalar_v<T>, int>>
std::optional<T> kitrt::envLookup(const std::string &var) {
  if (char *s = getenv(var.c_str()))
    return parseAs<T>(s, var);
  return std::nullopt;
}

// It is unlikely that we will ever want to parse a non-primitive type from
// an environment variable. To keep things clean, explicitly initialize all the
// types that we might need.
template std::optional<bool> kitrt::envLookup(const std::string &var);
template std::optional<int32_t> kitrt::envLookup(const std::string &var);
template std::optional<uint32_t> kitrt::envLookup(const std::string &var);
template std::optional<int64_t> kitrt::envLookup(const std::string &var);
template std::optional<uint64_t> kitrt::envLookup(const std::string &var);
template std::optional<float> kitrt::envLookup(const std::string &var);
template std::optional<double> kitrt::envLookup(const std::string &var);

void kitrt::envSet(const std::string &var, const std::string &val) {
  log(LABEL, "Setting in environment: %s=%s", var.c_str(), val.c_str());
  if (setenv(var.c_str(), val.c_str(), 1))
    warn(LABEL, "Could not set environment variable '%s'", var.c_str());
}

template <typename T, std::enable_if_t<std::is_scalar_v<T>, int>>
void kitrt::envSet(const std::string &var, const T &val) {
  kitrt::envSet(var, std::to_string(val));
}

template void kitrt::envSet(const std::string &var, const bool &);
template void kitrt::envSet(const std::string &var, const int32_t &);
template void kitrt::envSet(const std::string &var, const uint32_t &);
template void kitrt::envSet(const std::string &var, const int64_t &);
template void kitrt::envSet(const std::string &var, const uint64_t &);
template void kitrt::envSet(const std::string &var, const float &);
template void kitrt::envSet(const std::string &var, const double &);

void kitrt::envUnset(const std::string &var) {
  log(LABEL, "Unsetting in environment: %s", var.c_str());
  if (unsetenv(var.c_str()))
    warn(LABEL, "Could not unset environment variable '%s'", var.c_str());
}
