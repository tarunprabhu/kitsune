//===- KitsuneLoopAttrUtils.cpp - Utilities for Kitsune's loop attributes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities shared by Kitsune's loop attribute emitters
//
//===----------------------------------------------------------------------===//

#include "KitsuneLoopAttrUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

std::string getBaseName(const Record &attr) {
  auto addDot = [](char c, char prev) -> bool {
    return (std::isalpha(prev) && std::isdigit(c)) ||
           (std::isdigit(prev) && std::isalpha(c)) ||
           (std::islower(prev) && std::isupper(c));
  };

  std::string buf;
  raw_string_ostream os(buf);
  StringRef attrName = attr.getName();

  os << (char)std::tolower(attrName[0]);
  for (unsigned i = 1, ie = attrName.size(); i < ie; ++i) {
    if (addDot(attrName[i], attrName[i - 1]))
      os << ".";
    os << (char)std::tolower(attrName[i]);
  }

  os.flush();
  return buf;
}

std::string getIRName(const Record &attr) {
  std::string buf;
  raw_string_ostream os(buf);

  if (isTapirLoopOnly(attr))
    os << "tapir.";
  os << "loop." << getBaseName(attr);

  os.flush();
  return buf;
}

bool isTapirLoopOnly(const Record &attr) {
  const Record *allowedOn = attr.getValueAsDef("AllowedOn");
  return allowedOn->getName() == "TapirLoopsOnly";
}
