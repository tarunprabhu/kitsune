//===- ErrorHandling.cpp - Utilities for abnormal exits -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to deal with abnormal exists. These are slight variations on those
// provided by LLVM that are better suited for Kitsune.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;

void llvm::exitOnError() {
  sys::RunInterruptHandlers();
  std::exit(1);
}

void llvm::exitOnError(Error e) {
  raw_ostream &os = errs();
  bool hasColors = WithColor(os).colorsEnabled();

  WithColor::error(os);
  if (hasColors)
    os.changeColor(raw_ostream::SAVEDCOLOR, /*bold=*/true);
  os << toString(std::move(e));
  if (hasColors)
    os.resetColor();
  exitOnError();
}
