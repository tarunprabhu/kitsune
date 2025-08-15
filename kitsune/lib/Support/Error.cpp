//===- Error.cpp - Utilities to construct and report errors ---------------===//
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

#include "kitsune/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;

void llvm::detail::report_internal_error_impl(StringRef msg, StringRef file,
                                              unsigned line) {
  StringRef filename = sys::path::filename(file);
  std::string err = sjoin("INTERNAL ERROR at ", filename, ":", line, ": ", msg);
  WithColor::error();
  WithColor os(errs(), raw_ostream::SAVEDCOLOR, /*Bold=*/true);
  os << filename << ":" << line << ": " << msg << "\n";
  os.resetColor();
  report_fatal_error("INTERNAL COMPILER ERROR", /*gen_crash_diag=*/false);
}

void llvm::report_warning(StringRef msg) {
  WithColor::warning() << msg << "\n";
}

void llvm::report_error(Error e) {
  WithColor::error() << e << "\n";
  std::exit(2);
}

void llvm::report_error(StringRef msg) {
  WithColor::error() << msg << "\n";
  std::exit(2);
}

void llvm::report_error(const std::error_code &ec) {
  WithColor::error() << ec.message() << "\n";
  std::exit(2);
}

void llvm::report_error(const SMDiagnostic &diag) {
  WithColor::error();
  diag.print("", errs());
  std::exit(2);
}

void llvm::report_error(StringRef msg, const DebugLoc &loc) {
  llvm_unreachable("NOT IMPLEMENTED: report_error(DebugLoc)");
}
