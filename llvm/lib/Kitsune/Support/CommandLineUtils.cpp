//===--- CommandLine.cpp - Kitsune-specific shared command line options ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kitsune-specific command line options and utilities that are shared across
// tools.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/CommandLineUtils.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::Option &getOption(StringRef spelling) {
  StringMap<cl::Option *> clOpts = cl::getRegisteredOptions();
  assert(clOpts.count(spelling) && "Option must have been registered");

  return *clOpts[spelling];
}

void llvm::clSetOptionVisible(StringRef spelling) {
  getOption(spelling).setHiddenFlag(cl::NotHidden);
}

void llvm::clSetOptionDescription(StringRef spelling, StringRef descr) {
  getOption(spelling).setDescription(descr);
}
