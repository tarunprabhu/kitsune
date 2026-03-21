//===- KitAttrDocEmitter.cpp - Base class to emit attribute docs ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for emitters that generate documentation for Kitsune-specific
// attributes
//
//===----------------------------------------------------------------------===//

#include "KitAttrDocEmitter.h"
#include "KitAttrCommon.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

std::string KitAttrDocEmitter::getSectionLabel(const Record &attr) const {
  std::string name = getIRBaseName(attr);
  std::replace(name.begin(), name.end(), '.', '-');

  return name;
}

std::string KitAttrDocEmitter::quote(StringRef s, StringRef q) const {
  std::string buf;
  raw_string_ostream os(buf);

  os << q << s << q;
  os.flush();

  return buf;
}

std::string KitAttrDocEmitter::getEnum(const Record &attr) const {
  std::string buf;
  raw_string_ostream os(buf);

  os << "``" << getEnumName() << "::" << attr.getName() << "``";
  os.flush();

  return buf;
}

std::string KitAttrDocEmitter::getValueType(const Record &attr) const {
  StringRef valueType = getTypeName(attr);
  if (valueType.size())
    return quote(valueType, "``");
  return "";
}

void KitAttrDocEmitter::emitAttrHeader(raw_ostream &os, const Record &attr) {
  std::string irName = getIRName(getIRNamePrefix(attr), attr);
  os << ".. _" << getLabelPrefix() << "-attr-" << getSectionLabel(attr)
     << ":\n\n";
  os << irName << "\n";
  os << std::string(irName.size(), '-') << "\n";
  os << "\n";
}

void KitAttrDocEmitter::emitAttrArgs(raw_ostream &os, const Record &attr) {
  os << ".. csv-table::\n";
  os << "  :header: " << quote("Enum") << ", " << quote("Value Type") << "\n";
  os << "\n";
  os << "  " << getEnum(attr) << ", " << quote(getValueType(attr)) << "\n";
}

void KitAttrDocEmitter::emitAttrDoc(raw_ostream &os, const Record &attr) {
  os << attr.getValueAsString("Documentation") << "\n";
}

void KitAttrDocEmitter::run(raw_ostream &os) {
  const Record *globalDoc = records.getDef("GlobalDocumentation");
  if (!globalDoc)
    PrintFatalError("The GlobalDocumentation top-level definition is missing, "
                    "no documentation will be generated.");

  std::vector<const Record *> ordered;
  for (const Record *r : records.getAllDerivedDefinitions(getAttrBase()))
    ordered.push_back(r);
  std::sort(ordered.begin(), ordered.end(),
            [](const Record *l, const Record *r) -> bool {
              return l->getName() < r->getName();
            });

  os << globalDoc->getValueAsString("Intro");
  for (const Record *attr : ordered) {
    os << "\n";
    emitAttrHeader(os, *attr);
    emitAttrArgs(os, *attr);
    os << "\n";
    emitAttrDoc(os, *attr);
  }
}

KitAttrDocEmitter::KitAttrDocEmitter(const RecordKeeper &records)
    : records(records) {}
