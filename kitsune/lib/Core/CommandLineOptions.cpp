//===- CommandLineOptions.cpp - Command line options for Kitsune ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of core command line options for Kitsune's tools.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/CommandLineOptions.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

cl::OptionCategory llvm::catKitCommon("Kitsune Options (common)");
cl::OptionCategory llvm::catKitCore("Kitsune Options (core)");
cl::OptionCategory llvm::catKitDev("Kitsune Developer Options");

// Although a larger number of tapir targets are present within the repo,
// several are either known to be non-functional, in a severe state of
// disrepair, or have not been tested to ensure that they work as expected with
// the rest of Kitsune (this is the case for tapir targets that have been
// imported from the OpenCilk compiler). To avoid any unnecessary "surprises",
// we only allow those tapir targets to be set that have received a "reasonable"
// amount of attention.
cl::opt<TTID>
    llvm::clTapir("tapir", cl::desc("The primary tapir target"),
                  cl::init(TTID::Nolo), cl::value_desc("target"),
                  cl::cat(catKitCommon),
                  cl::values(clEnumValN(TTID::Nolo, "nolo", ""),
                             clEnumValN(TTID::Serial, "serial", ""),
                             clEnumValN(TTID::Cuda, "cuda", ""),
                             clEnumValN(TTID::Hip, "hip", ""),
                             clEnumValN(TTID::OpenCilk, "opencilk", "")));

void llvm::clSetOptionDescription(StringRef spelling, StringRef descr) {
  StringMap<cl::Option *> clOpts = cl::getRegisteredOptions();
  assert(clOpts.count(spelling) && "Option must have been registered");

  clOpts[spelling]->setDescription(descr);
}
