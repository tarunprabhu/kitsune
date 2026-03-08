//===- KitsuneAttrUtils.cpp - Utilities for Kitsune's attributes ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities shared by Kitsune's attribute emitters.
//
//===----------------------------------------------------------------------===//

#include "KitsuneAttrUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

std::string getBaseName(const Record &attr) {
  auto addDot = [](char c, char prev) -> bool {
    return (isAlpha(prev) && isDigit(c)) || (isDigit(prev) && isAlpha(c)) ||
           (isLower(prev) && isUpper(c));
  };

  std::string buf;
  raw_string_ostream os(buf);
  StringRef attrName = attr.getName();

  os << (char)toLower(attrName[0]);
  for (unsigned i = 1, ie = attrName.size(); i < ie; ++i) {
    if (addDot(attrName[i], attrName[i - 1]))
      os << ".";
    os << (char)toLower(attrName[i]);
  }

  os.flush();
  return buf;
}
