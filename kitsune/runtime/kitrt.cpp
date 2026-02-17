//===- kitrt.cpp - Routines common to several of Kitsune's runtimes -------===//
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

#include "kitrt.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <execinfo.h>
#include <type_traits>

// FIXME: Combine these global variables into a single struct. This should
// also be private. However, we expose it because it is examined often by
// most runtimes to determine whether to print informational messages. Wrapping
// it in a function may be expensive.
bool _kitrt_verbose_mode = false;

#ifdef __cplusplus
extern "C" {
#endif

void __kitrt_enable_verbose_mode() { _kitrt_verbose_mode = true; }

void __kitrt_disable_verbose_mode() { _kitrt_verbose_mode = false; }

void __kitrt_set_verbose_mode(bool enable) { _kitrt_verbose_mode = enable; }

void __kitrt_initialize() {
  (void)__kitrt_get_env_value("KITRT_VERBOSE", _kitrt_verbose_mode);
  __kitrt_message("kitrt", "verbose mode enabled from environment variable");
}

// Write a message to stderr. \p category is optional. If \p is a format string,
// the variable list of arguments \p args must be of the appropriate types.
static void __kitrt_log(const char *label, const char *category,
                        const char *msg, va_list args) {
  // TODO: It would be nice if we could colorize the label.
  if (label)
    fprintf(stderr, "%s: ", label);
  if (category)
    fprintf(stderr, "%s: ", category);
  vfprintf(stderr, msg, args);
  fprintf(stderr, "\n");
}

static void __kitrt_log(const char *label, const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  __kitrt_log(label, nullptr, msg, args);
  va_end(args);
}

[[noreturn]] void __kitrt_fatal(const char *label, const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  __kitrt_log(label, "ERROR", msg, args);
  va_end(args);

  std::exit(EXIT_FAILURE);
}

void __kitrt_error(const char *label, const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  __kitrt_log(label, "ERROR", msg, args);
  va_end(args);
}

void __kitrt_warn(const char *label, const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  __kitrt_log(label, "WARNING", msg, args);
  va_end(args);
}

void __kitrt_message(const char *label, const char *msg, ...) {
  if (__kitrt_verbose_mode()) {
    va_list args;
    va_start(args, msg);
    __kitrt_log(label, nullptr, msg, args);
    va_end(args);
  }
}

void __kitrt_print_stack_trace(void) {
  const unsigned depth = 25;
  void *trace[depth];
  int size = backtrace(trace, depth);
  if (char **strings = backtrace_symbols(trace, size)) {
    __kitrt_log("kitrt", "stack trace (%d frames)", size);
    for (int i = 0; i < size; i++)
      __kitrt_log("kitrt", "  %s", strings[i]);
    __kitrt_log("kitrt", "end stack trace");
    free(strings);
  }
}

void __kitrt_set_env(const char *varname, const char *value) {
  assert(varname && "Missing variable name");
  assert(value && "Missing value destination");
  if (setenv(varname, value, 0))
    __kitrt_warn("kitrt", "could not set environment variable '%s'", varname);
}

void __kitrt_unset_env(const char *varname) {
  assert(varname && "Missing variable name");
  if (unsetenv(varname))
    __kitrt_warn("kitrt", "could not unset environment variable '%s'", varname);
}

#ifdef __cplusplus
} // extern "C"
#endif

template <typename T>
static bool parse(const char *varname, const char *str, T &out);

template <> bool parse(const char *varname, const char *str, int &out) {
  out = atoi(str);
  return true;
}

template <> bool parse(const char *varname, const char *str, unsigned &out) {
  out = atoi(str);
  return true;
}

template <> bool parse(const char *varname, const char *str, bool &out) {
  char *lstr = strdup(str);
  for (int i = 0; str[i]; i++)
    lstr[i] = tolower(str[i]);

  if (!strcmp(lstr, "true") || !strcmp(lstr, "1")) {
    out = true;
  } else if (!strcmp(lstr, "false") || !strcmp(lstr, "0")) {
    out = false;
  } else {
    __kitrt_warn("kitrt",
                 "environment variable '%s' not set to 'true' or 'false'. "
                 "Assuming 'true'",
                 varname);
    out = true;
  }
  return true;
}

template <> bool parse(const char *varname, const char *str, long &out) {
  out = atol(str);
  return true;
}

template <>
bool parse(const char *varname, const char *str, unsigned long &out) {
  out = atol(str);
  return true;
}

template <> bool parse(const char *varname, const char *str, long long &out) {
  out = atoll(str);
  return true;
}

template <>
bool parse(const char *varname, const char *str, unsigned long long &out) {
  out = atoll(str);
  return true;
}

template <> bool parse(const char *varname, const char *str, float &out) {
  out = (float)atof(str);
  return true;
}

template <> bool parse(const char *varname, const char *str, double &out) {
  out = atof(str);
  return true;
}

template <typename ValueType>
bool __kitrt_get_env_value(const char *varname, ValueType &value) {
  assert(varname && "Expected variable name");

  if (char *vstr = getenv(varname)) {
    if constexpr (std::is_same_v<ValueType, int>)
      return parse<int>(varname, vstr, value);
    else if constexpr (std::is_same_v<ValueType, unsigned>)
      return parse<unsigned>(varname, vstr, value);
    else if constexpr (std::is_same_v<ValueType, bool>)
      return parse<bool>(varname, vstr, value);
    else if constexpr (std::is_same_v<ValueType, long>)
      return parse<long>(varname, vstr, value);
    else if constexpr (std::is_same_v<ValueType, unsigned long>)
      return parse<unsigned long>(varname, vstr, value);
    else if constexpr (std::is_same_v<ValueType, long long>)
      return parse<long long>(varname, vstr, value);
    else if constexpr (std::is_same_v<ValueType, unsigned long long>)
      return parse<unsigned long long>(varname, vstr, value);
    else if constexpr (std::is_same_v<ValueType, float>)
      return parse<float>(varname, vstr, value);
    else if constexpr (std::is_same_v<ValueType, double>)
      return parse<double>(varname, vstr, value);
    else
      static_assert(0 && "No registered environment variable parser for type");
  }

  return false;
}

// It is unlikely that we will ever want to parse a non-primitive type from
// an environment variable. To keep things clean, explicitly initialize all the
// types that we might care about.
template bool __kitrt_get_env_value(const char *var, bool &);
template bool __kitrt_get_env_value(const char *var, int &);
template bool __kitrt_get_env_value(const char *var, unsigned &);
template bool __kitrt_get_env_value(const char *var, long &);
template bool __kitrt_get_env_value(const char *var, unsigned long &);
template bool __kitrt_get_env_value(const char *var, long long &);
template bool __kitrt_get_env_value(const char *var, unsigned long long &);
template bool __kitrt_get_env_value(const char *var, float &);
template bool __kitrt_get_env_value(const char *var, double &);
