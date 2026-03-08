//===- KitsuneLoopAttrUtils.cpp - Utilities for Kitsune's loop attributes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities shared by Kitsune's loop attribute emitters.
//
//===----------------------------------------------------------------------===//

#include "KitsuneLoopAttrUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

std::string getIRName(const Record &attr) {
  SmallString<32> buf;
  raw_svector_ostream os(buf);

  if (isTapirLoopOnly(attr))
    os << "tapir.";
  os << "loop." << getBaseName(attr);

  return buf.c_str();
}

bool isTapirLoopOnly(const Record &attr) {
  const Record *allowedOn = attr.getValueAsDef("AllowedOn");
  return allowedOn->getName() == "TapirLoopsOnly";
}
