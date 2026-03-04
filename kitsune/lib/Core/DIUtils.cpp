//===- DIUtils.cpp - Utilities for DebugInfo ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to work with debug information.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/DIUtils.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

std::string llvm::toString(const DebugLoc &dbgLoc, bool inlinedAt) {
  if (!dbgLoc)
    return "";

  StringRef file = cast<DIScope>(dbgLoc.getScope())->getFilename();
  if (file.empty())
    return "";

  std::string buf;
  raw_string_ostream os(buf);

  os << file << ":" << dbgLoc.getLine();
  if (unsigned col = dbgLoc.getCol())
    os << ":" << col;

  if (inlinedAt) {
    std::string inlined = toString(dbgLoc.getInlinedAt(), /*inlinedAt*/ false);
    if (inlined.size())
      os << "@[" << inlined << "]";
  }

  os.flush();
  return buf;
}
