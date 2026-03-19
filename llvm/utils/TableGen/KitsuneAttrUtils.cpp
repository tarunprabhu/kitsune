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

static std::string getName(StringRef base, const Record &attr) {
  auto addDot = [](char c, char prev) -> bool {
    return (isAlpha(prev) && isDigit(c)) || (isDigit(prev) && isAlpha(c)) ||
           (isLower(prev) && isUpper(c));
  };

  std::string buf;
  raw_string_ostream os(buf);
  StringRef attrName = attr.getName();

  os << base;
  os << (char)toLower(attrName[0]);
  for (unsigned i = 1, ie = attrName.size(); i < ie; ++i) {
    if (addDot(attrName[i], attrName[i - 1]))
      os << ".";
    os << (char)toLower(attrName[i]);
  }
  os.flush();

  return buf;
}

std::string getAttrBaseName(const Record &attr) {
  return getName("", attr);
}

std::string getInstAttrIRName(const Record &attr) {
  return getName("kit.inst.", attr);
}

std::string getLoopAttrIRName(const Record &attr) {
  if (isTapirLoopOnly(attr))
    return getName("tapir.loop.", attr);
  return getName("loop.", attr);
}

std::string getModuleAttrIRName(const Record& attr) {
  return getName("kit.module.", attr);
}

bool isTapirLoopOnly(const Record &attr) {
  const Record *allowedOn = attr.getValueAsDef("AllowedOn");
  return allowedOn->getName() == "TapirLoopsOnly";
}
