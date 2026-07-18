//===- logging.cpp - Logging utilites for Kitsune's runtime ---------------===//
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

#include "common/logging.h"
#include "common/env.h"
#include "global/global.h"
#include "kitrt.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>

using namespace kitrt;

// Write a message to stderr. \p category is optional. If \p is a format string,
// the variable list of arguments \p args must be of the appropriate types.
static void logImpl(const char *color, const char *tag, const char *category,
                    const char *msg, va_list args) {
  static std::mutex mtx;
  std::lock_guard<std::mutex> guard(mtx);

  fprintf(stderr, "kitrt: ");
  if (!gctx.colors) {
    if (tag)
      fprintf(stderr, "[%s]: ", tag);
    if (category)
      fprintf(stderr, "%s: ", category);
    vfprintf(stderr, msg, args);
  } else if (category) {
    // If a category is set, and colors are enabled, a tag will be printed in
    // the default color for the tag. This will be followed by the category in
    // boldface and a color depending on the category itself. Finally, the
    // message will be printed in bold in the default color of the terminal.
    //
    // The HTML-ish code below is an approximation of the output.
    //
    // <color>[$tag]</color>: <b><catColor>$category</catColor>: $msg</b>
    //
    if (tag)
      fprintf(stderr, "%s[%s]%s: ", color ? color : "", tag, ANSI_RESET);
    const char *categoryColor = ANSI_RESET;
    if (strcmp(category, "error") == 0)
      categoryColor = ANSI_RED;
    else if (strcmp(category, "warning") == 0)
      categoryColor = ANSI_MAGENTA;
    fprintf(stderr, "%s%s%s%s", categoryColor, ANSI_BOLD, category, ANSI_RESET);
    fprintf(stderr, "%s: ", ANSI_BOLD);
    vfprintf(stderr, msg, args);
    fprintf(stderr, "%s", ANSI_RESET);
  } else {
    // If a category is not set an colors are enabled, the tag and message will
    // all be printed in the same color.
    fprintf(stderr, "%s", color ? color : "");
    if (tag)
      fprintf(stderr, "[%s]: ", tag);
    vfprintf(stderr, msg, args);
    fprintf(stderr, "%s", color ? ANSI_RESET : "");
  }
  fprintf(stderr, "\n");
}

[[noreturn]] void kitrt::fatal(const char *color, const char *tag,
                               const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  logImpl(color, tag, "error", msg, args);
  va_end(args);

  std::exit(EXIT_FAILURE);
}

void kitrt::error(const char *color, const char *tag, const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  logImpl(color, tag, "error", msg, args);
  va_end(args);
}

void kitrt::warn(const char *color, const char *tag, const char *msg, ...) {
  va_list args;
  va_start(args, msg);
  logImpl(color, tag, "warning", msg, args);
  va_end(args);
}

void kitrt::log(const char *color, const char *tag, const char *msg, ...) {
  if (gctx.verbose) {
    va_list args;
    va_start(args, msg);
    logImpl(color, tag, nullptr, msg, args);
    va_end(args);
  }
}

void kitrt::logIfVerbose(const char *tag, const char *msg, ...) {
  if (envLookupOr(envVerbose, envVerboseLegacy, false)) {
    va_list args;
    va_start(args, msg);
    logImpl(nullptr, tag, nullptr, msg, args);
    va_end(args);
  }
}
