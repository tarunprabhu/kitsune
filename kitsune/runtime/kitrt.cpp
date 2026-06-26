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

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <execinfo.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#define LABEL "kitrt"

// FIXME: Combine all global variables here into a single struct.
static bool __kitrt_initialized = false;
static bool __kitrt_finalized = false;

// This should be private. However, we expose it because it is examined often by
// most runtimes to determine whether to print informational messages. Wrapping
// it in a function may be expensive.
bool _kitrt_verbose_mode = false;

extern "C" void __kitrt_enable_verbose_mode() { _kitrt_verbose_mode = true; }

extern "C" void __kitrt_disable_verbose_mode() { _kitrt_verbose_mode = false; }

extern "C" void __kitrt_set_verbose_mode(bool enable) {
  _kitrt_verbose_mode = enable;
}

extern "C" void __kitrt_initialize() {
  if (__kitrt_initialized)
    return;

  (void)__kitrt_get_env_value("KITRT_VERBOSE", _kitrt_verbose_mode);
  if (!_kitrt_verbose_mode)
    (void)__kitrt_get_env_value("KIT_VERBOSE", _kitrt_verbose_mode);

  // This really ought to be the first thing in this function, but if we move
  // it before the KIT_VERBOSE environment variable is read, it may not be
  // printed. It may be confusing for users if they see the
  // "initializing ... initialized" pattern everywhere, except for the common
  // initialization.
  __kitrt_message(LABEL, "Initializing Kitsune runtime (common)");

  // This message will only be printed if verbose mode is actually set.
  __kitrt_message(LABEL, "Verbose mode enabled");

  __kittimer_initialize();
#ifdef KITRT_PAPI_ENABLED
  __kitpapi_initialize();
#endif // KITRT_PAPI_ENABLED

  __kitrt_initialized = true;
  __kitrt_message(LABEL, "Initialized Kitsune runtime (common)");
}

extern "C" void __kitrt_finalize() {
  if (__kitrt_finalized)
    return;

  __kitrt_message(LABEL, "Finalizing Kitsune runtime (common)");

#ifdef KITRT_PAPI_ENABLED
  __kitpapi_finalize();
#endif // KITRT_PAPI_ENABLED
  __kittimer_finalize();

  __kitrt_finalized = true;
  __kitrt_message(LABEL, "Finalized Kitsune runtime (common)");
}

// Write a message to stderr. \p category is optional. If \p is a format string,
// the variable list of arguments \p args must be of the appropriate types.
static void __kitrt_log(const char *label, const char *category, bool newline,
                        const char *msg, va_list args) {
  static std::mutex mtx;
  std::lock_guard<std::mutex> guard(mtx);

  // TODO: It would be nice if we could colorize the label.
  if (label)
    fprintf(stderr, "%s: ", label);
  if (category)
    fprintf(stderr, "%s: ", category);
  vfprintf(stderr, msg, args);
  if (newline)
    fprintf(stderr, "\n");
}

static void __kitrt_log(const char *label, bool newline, const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  __kitrt_log(label, nullptr, newline, msg, args);
  va_end(args);
}

[[noreturn]] void __kitrt_fatal(const char *label, const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  __kitrt_log(label, "ERROR", true, msg, args);
  va_end(args);

  std::exit(EXIT_FAILURE);
}

void __kitrt_error(const char *label, const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  __kitrt_log(label, "ERROR", true, msg, args);
  va_end(args);
}

void __kitrt_warn(const char *label, const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  __kitrt_log(label, "WARNING", true, msg, args);
  va_end(args);
}

void __kitrt_message(const char *label, const char *msg, ...) {
  if (__kitrt_verbose_mode()) {
    va_list args;
    va_start(args, msg);
    __kitrt_log(label, nullptr, true, msg, args);
    va_end(args);
  }
}

void __kitrt_error_noflush(const char *label, const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  __kitrt_log(label, "ERROR", false, msg, args);
  va_end(args);
}

void __kitrt_warn_noflush(const char *label, const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  __kitrt_log(label, "WARNING", false, msg, args);
  va_end(args);
}

void __kitrt_message_noflush(const char *label, const char *msg, ...) {
  if (__kitrt_verbose_mode()) {
    va_list args;
    va_start(args, msg);
    __kitrt_log(label, nullptr, false, msg, args);
    va_end(args);
  }
}

extern "C" void __kitrt_print_stack_trace(void) {
  const unsigned depth = 25;
  void *trace[depth];
  int size = backtrace(trace, depth);
  if (char **strings = backtrace_symbols(trace, size)) {
    __kitrt_log("kitrt", true, "stack trace (%d frames)", size);
    for (int i = 0; i < size; i++)
      __kitrt_log("kitrt", true, "  %s", strings[i]);
    __kitrt_log("kitrt", true, "end stack trace");
    free(strings);
  }
}

extern "C" void __kitrt_set_env(const char *varname, const char *value) {
  assert(varname && "Missing variable name");
  assert(value && "Missing value destination");
  if (setenv(varname, value, 0))
    __kitrt_warn("kitrt", "could not set environment variable '%s'", varname);
}

extern "C" void __kitrt_unset_env(const char *varname) {
  assert(varname && "Missing variable name");
  if (unsetenv(varname))
    __kitrt_warn("kitrt", "could not unset environment variable '%s'", varname);
}

unsigned nearestPowerOf2LE(unsigned n) {
  unsigned p = 1;
  while (p <= n)
    p <<= 1;
  return p >> 1;
}

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
    __kitrt_warn("kitrt",
                 "ignoring environment variable '%s'. Values contains unparsed "
                 "characters",
                 vname);
  } catch (std::invalid_argument) {
    __kitrt_warn("kitrt",
                 "ignoring environment variable '%s'. Value is not valid",
                 vname);
  } catch (std::out_of_range) {
    __kitrt_warn("kitrt",
                 "ignoring environment variable '%s'. Value is not in range",
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
    __kitrt_warn("kitrt",
                 "environment variable '%s' not set to 'true' or 'false'. "
                 "Assuming 'true'",
                 vname);
    v = true;
  }
  return true;
}

template <> bool parseInto(int &v, const std::string &vstr, const char *vname) {
  using Converter = int(const std::string &, std::size_t *, int);
  return parseInto(v, vstr, vname, (Converter *)&std::stoi, /*base=*/10);
}

template <>
bool parseInto(unsigned &v, const std::string &vstr, const char *vname) {
  using Converter = int(const std::string &, std::size_t *, int);
  return parseInto(v, vstr, vname, (Converter *)&std::stoi, /*base=*/10);
}

template <>
bool parseInto(long &v, const std::string &vstr, const char *vname) {
  using Converter = long(const std::string &, std::size_t *, int);
  return parseInto(v, vstr, vname, (Converter *)&std::stol, /*base=*/10);
}

template <>
bool parseInto(unsigned long &v, const std::string &vstr, const char *vname) {
  using Converter = unsigned long(const std::string &, std::size_t *, int);
  return parseInto(v, vstr, vname, (Converter *)&std::stoul, /*base=*/10);
}

template <>
bool parseInto(long long &v, const std::string &vstr, const char *vname) {
  using Converter = long long(const std::string &, std::size_t *, int);
  return parseInto(v, vstr, vname, (Converter *)&std::stoll, /*base=*/10);
}

template <>
bool parseInto(unsigned long long &v, const std::string &vstr,
               const char *vname) {
  using Converter = unsigned long long(const std::string &, std::size_t *, int);
  return parseInto(v, vstr, vname, (Converter *)&std::stoull, /*base=*/10);
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

template <typename V> bool __kitrt_get_env_value(const char *vname, V &v) {
  assert(vname && "Expected variable name");

  if (char *vstr = getenv(vname))
    return parseInto<V>(v, vstr, vname);
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

extern "C" unsigned __kitrt_num_threads_from_env() {
  const char *envNumThreads = getenv(__kitrt_envname_num_threads);
  if (!envNumThreads) {
    __kitrt_message(LABEL, "Environment variable '%s' not set",
                    __kitrt_envname_num_threads);
    return 0;
  }

  __kitrt_message(LABEL, "Environment variable %s=%s",
                  __kitrt_envname_num_threads, envNumThreads);
  char *end = nullptr;
  long numThreads = strtol(envNumThreads, &end, 10);

  // If *end is not '\0', either there are additional non-numeric characters in
  // the environment variable, or the environment variable cannot be parsed to a
  // signed integer. In this case, ignore the environment variable. If the value
  // value of the environment variable is a valid integer, it may yet be out of
  // range, in which case errno will be set to ERANGE.
  bool error = *end != '\0' || errno == ERANGE || numThreads < 0 ||
               numThreads > std::numeric_limits<int>::max();
  if (error) {
    __kitrt_warn(LABEL, "Invalid number of threads in %s",
                 __kitrt_envname_num_threads);
    return 0;
  }

  __kitrt_message(LABEL, "Number of threads = %d", numThreads);
  return numThreads;
}

extern "C" unsigned __kitrt_num_cpus() {
  __kitrt_message(LABEL, "Determining number of CPUs");

  // TODO: Does this work as expected on all platforms? It seems to for the
  // platforms that we care about, but it may be better to use something more
  // reliable instead.
  unsigned cpus = std::thread::hardware_concurrency();
  if (cpus == 0) {
    __kitrt_warn(LABEL, "Could not determine number of CPUs");
    return 1;
  }

  __kitrt_message(LABEL, "Found %d CPUs", cpus);
  return cpus;
}
