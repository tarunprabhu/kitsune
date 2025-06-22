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

#include "kitsune/Support/CommandLine.h"

using namespace llvm;

namespace llvm {

/// Option category for kitsune options hat are shared by several tools.
static cl::OptionCategory catKitClOpts("Kitsune Options");

/// Command line option to specify a tapir target. In some tools, the purpose of
/// this option may be used to mean something other than what the description
/// here states.
///
/// This is only a subset of all the known tapir target ID's. This is
/// intentional. At the time of writing, the TTID's enum includes all
/// the tapir targets that are known - even if the code for those targets has
/// completely bit-rotted, or if it has not been fully implemented. That is
/// not terribly useful. This only contains those tapir targets that actually
/// have full implementations.
static cl::opt<TTID>
    clTapir("tapir", cl::desc("The primary tapir target"), cl::init(TTID::None),
            cl::cat(catKitClOpts),
            cl::values(clEnumValN(TTID::None, "none", ""),
                       clEnumValN(TTID::Serial, "serial", ""),
                       clEnumValN(TTID::Cuda, "cuda", ""),
                       clEnumValN(TTID::Hip, "hip", ""),
                       clEnumValN(TTID::OpenCilk, "opencilk", "")));

/// This was the option originally in tapir, but in Kitsune, we prefer to use
/// --tapir instead.
static cl::alias clTapirTarget("tapir-target", cl::desc("Alias for --tapir"),
                               cl::aliasopt(clTapir), cl::cat(catKitClOpts));

} // namespace llvm

cl::OptionCategory &llvm::getKitClOptCategory() { return catKitClOpts; }

std::optional<TTID> llvm::getClOptTapir(std::optional<TTID> defawlt) {
  if (clTapir.getNumOccurrences())
    return clTapir;
  return defawlt;
}
