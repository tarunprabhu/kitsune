//===- TapirTargetOptions.cpp - Options for the tapir targets -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared implementation for the base TapirTargetOptions object. Also contains
// any command line options shared by all tapir targets.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/TapirTargetOptions.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

static cl::opt<bool>
    clTapirVerbose("tapir-verbose", cl::init(false), cl::NotHidden,
                   cl::desc("Enable verbose mode in all tapir targets"));

static cl::opt<bool>
    clKitrtVerbose("kitrt-verbose", cl::init(false), cl::NotHidden,
                   cl::desc("Enable verbose mode in kitsune's runtime"));

void TapirTargetOptions::readClOptions() {
  this->verbose = clTapirVerbose;
  this->runtimeVerbose = clTapirVerbose or clKitrtVerbose;
}

raw_ostream &operator<<(raw_ostream &os, const bool &v) {
  return os << (v ? "true" : "false");
}

} // namespace llvm
