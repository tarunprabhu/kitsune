//===- MaybeBool.cpp - Value that may be a boolean or unset ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Enumeration for a value that is either a boolean or unset.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/MaybeBool.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

template <> std::string llvm::toString(const MaybeBool &v) {
  switch (v) {
  case MaybeBool::Off: return "off";
  case MaybeBool::On: return "on";
  case MaybeBool::Any: return "any";
  }
  llvm_unreachable("toString: MaybeBool value not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const MaybeBool &v) {
  os << toString(v);
  return os;
}

template <> std::optional<MaybeBool> llvm::fromString(StringRef s) {
  return StringSwitch<std::optional<MaybeBool>>(s)
      .Case("off", MaybeBool::Off)
      .Case("on", MaybeBool::On)
      .Case("any", MaybeBool::Any)
      .Default(std::nullopt);
}
