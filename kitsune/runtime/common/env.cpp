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
#include <cassert>
#include <cstdint>
#include <ctype.h>
#include <limits>
#include <stdexcept>
#include <string>

#define LABEL "kitrt"

template <typename F, typename V, typename... Args>
static bool parseInto(V &out, const std::string &vstr, const char *vname,
                      F converter, Args &&...args) {
  try {
    std::size_t pos = 0;
    auto tmp = converter(vstr, &pos, args...);
    if (pos == vstr.size()) {
      out = tmp;
      return true;
    }
    __kitrt_warn(LABEL,
                 "Ignoring environment variable '%s'. Values contains unparsed "
                 "characters",
                 vname);
  } catch (std::invalid_argument) {
    __kitrt_warn(
        LABEL, "Ignoring environment variable '%s'. Value is not valid", vname);
  } catch (std::out_of_range) {
    __kitrt_warn(LABEL,
                 "Ignoring environment variable '%s'. Value is not in range",
                 vname);
  }
  return false;
}

template <typename V>
static bool parseInto(V &v, const std::string &str, const char *vname);

template <>
bool parseInto(bool &v, const std::string &vstr, const char *vname) {
  auto equals = [](const std::string &l, const std::string &r) -> bool {
    return std::equal(l.begin(), l.end(), r.begin(),
                      [](unsigned char cl, unsigned char cr) -> bool {
                        return std::tolower(cl) == std::tolower(cr);
                      });
  };

  if (equals(vstr, "true") || vstr == "1") {
    v = true;
  } else if (equals(vstr, "false") || vstr == "0") {
    v = false;
  } else {
    // FIXME: We should be more strict and reject this, but for now, we are
    // permissive.
    __kitrt_warn(LABEL,
                 "Environment variable '%s' not set to 'true' or 'false'. "
                 "Assuming 'true'",
                 vname);
    v = true;
  }
  return true;
}

template <>
bool parseInto(int32_t &v, const std::string &vstr, const char *vname) {
  using Converter = int32_t(const std::string &, std::size_t *, int);
  return parseInto(v, vstr, vname, (Converter *)&std::stoi, /*base=*/10);
}

template <>
bool parseInto(uint32_t &v, const std::string &vstr, const char *vname) {
  using Converter = uint64_t(const std::string &, std::size_t *, int);
  uint64_t tmp;
  bool ok = parseInto(tmp, vstr, vname, (Converter *)&std::stoul, /*base=*/10);
  if (!ok || tmp > std::numeric_limits<uint32_t>::max())
    return false;
  v = tmp;
  return true;
}

template <>
bool parseInto(int64_t &v, const std::string &vstr, const char *vname) {
  using Converter = int64_t(const std::string &, std::size_t *, int);
  return parseInto(v, vstr, vname, (Converter *)&std::stol, /*base=*/10);
}

template <>
bool parseInto(uint64_t &v, const std::string &vstr, const char *vname) {
  using Converter = uint64_t(const std::string &, std::size_t *, int);
  return parseInto(v, vstr, vname, (Converter *)&std::stoul, /*base=*/10);
}

template <>
bool parseInto(float &v, const std::string &vstr, const char *vname) {
  using Converter = float(const std::string &, std::size_t *);
  return parseInto(v, vstr, vname, (Converter *)&std::stof);
}

template <>
bool parseInto(double &v, const std::string &vstr, const char *vname) {
  using Converter = double(const std::string &, std::size_t *);
  return parseInto(v, vstr, vname, (Converter *)&std::stod);
}

template <typename V> bool __kitrt_env_lookup(const char *vname, V &v) {
  assert(vname && "Expected variable name");

  if (char *vstr = getenv(vname))
    return parseInto<V>(v, vstr, vname);
  return false;
}

// It is unlikely that we will ever want to parse a non-primitive type from
// an environment variable. To keep things clean, explicitly initialize all the
// types that we might need.
template bool __kitrt_env_lookup(const char *var, bool &);
template bool __kitrt_env_lookup(const char *var, int32_t &);
template bool __kitrt_env_lookup(const char *var, uint32_t &);
template bool __kitrt_env_lookup(const char *var, int64_t &);
template bool __kitrt_env_lookup(const char *var, uint64_t &);
template bool __kitrt_env_lookup(const char *var, float &);
template bool __kitrt_env_lookup(const char *var, double &);

void __kitrt_env_set(const char *varname, const char *s) {
  assert(varname && "Missing variable name");

  __kitrt_message(LABEL, "Setting in environment: %s=%s", varname, s);
  if (setenv(varname, s, 1))
    __kitrt_warn(LABEL, "Could not set environment variable '%s'", varname);
}

template <typename T, std::enable_if_t<std::is_scalar_v<T>, int>>
void __kitrt_env_set(const char *varname, const T &value) {
  std::string s = std::to_string(value);
  ::__kitrt_env_set(varname, s.c_str());
}

template void __kitrt_env_set(const char *var, const bool &);
template void __kitrt_env_set(const char *var, const int32_t &);
template void __kitrt_env_set(const char *var, const uint32_t &);
template void __kitrt_env_set(const char *var, const int64_t &);
template void __kitrt_env_set(const char *var, const uint64_t &);
template void __kitrt_env_set(const char *var, const float &);
template void __kitrt_env_set(const char *var, const double &);

void __kitrt_env_unset(const char *varname) {
  assert(varname && "Missing variable name");
  __kitrt_message(LABEL, "Unsetting in environment: %s", varname);
  if (unsetenv(varname))
    __kitrt_warn(LABEL, "Could not unset environment variable '%s'", varname);
}
