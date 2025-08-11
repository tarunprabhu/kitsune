//===- Error.h - Utilities to construct and report errors ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to construct and report errors. Unlike clang/flang, we may only
// catch some errors intended for users in LLVM. Printing diagnostics here is
// not the most ergonomic, so we provide some utilities to make it easier.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_ERROR_H
#define KITSUNE_SUPPORT_ERROR_H

#include "kitsune/Support/StringUtils.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class DebugLoc;
class SMDiagnostic;

namespace detail {

/// Print a message to stderr indicating that an internal compiler error
/// occurred and exit immediately.
void report_internal_error_impl(StringRef msg, StringRef file = "",
                                unsigned line = 0);

} // namespace detail

// Report a fatal internal compiler error and exit immediately.
#if defined(__FILE__) && defined(__LINE__)
#define report_internal_error(...)                                             \
  do {                                                                         \
    std::string err = sjoin(__VA_ARGS__);                                      \
    detail::report_internal_error_impl(err, __FILE__, __LINE__);               \
  } while (0)
#else
#define report_internal_error(...)                                             \
  do {                                                                         \
    std::string err = sjoin(__VA_ARGS__);                                      \
    detail::report_internal_error_impl(err);                                   \
  } while (0)
#endif // defined(__FILE__) && defined(__LINE__)

/// Print a warning message.
void report_warning(StringRef msg);

/// Print an error message and exit immediately. This is a "cleaner" way to exit
/// on an error than \ref report_fatal_error and is intended to be used to
/// report "user" errors. \ref report_fatal_error should be used to report
/// internal compiler errors.
void report_error(StringRef msg);

/// Print an error message and exit immediately.
void report_error(const std::error_code &ec);

/// Print the error message in the given diagnostic object and exit immediately.
void report_error(const SMDiagnostic &sm);

/// Print an error message and exit immediately. This should be used to report
/// "delayed" errors. These are errors that cannot be caught during semantic
/// checking, or the result of a static analysis that operates on LLVM-IR.
void report_error(StringRef msg, const DebugLoc &loc);

/// Print an error message and exit immediately. This just calls LLVM's
/// llvm_fatal_error internally, but generates a more user-friendly error
/// message. This is intended to display an error when an internal compiler
/// error is encountered. "User" errors should be reported with \ref
/// report_error.
void report_fatal_error(StringRef msg);

} // namespace llvm

#endif // KITSUNE_SUPPORT_ERROR_H
