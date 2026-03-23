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

std::string KitAttrDocEmitter::getEnum(const Record &attr) const {
  std::string buf;
  raw_string_ostream os(buf);

  os << "``" << getEnumName() << "::" << attr.getName() << "``";
  os.flush();

  return buf;
}

std::string KitAttrDocEmitter::getValueType(const Record &attr) const {
  StringRef valueType = getBasicTypeName(attr);
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

static raw_ostream &pad(raw_ostream &os, StringRef s, size_t w) {
  size_t l = s.size();
  size_t lp = (w - l) / 2;
  size_t rp = w - l - lp;
  for (unsigned i = 0; i < lp; ++i)
    os << " ";
  os << s;
  for (unsigned i = 0; i < rp; ++i)
    os << " ";
  return os;
}

static raw_ostream &line(raw_ostream &os, StringRef s1, size_t w1, StringRef s2,
                         size_t w2, StringRef s3, size_t w3) {
  os << "| ";
  pad(os, s1, w1) << " | ";
  pad(os, s2, w2) << " | ";
  pad(os, s3, w3);
  os << " |\n";

  return os;
}

static raw_ostream &sep(raw_ostream &os, char c, StringRef s1, StringRef s2,
                        StringRef s3) {
  os << "+" << c << s1;
  os << c << "+" << c << s2;
  os << c << "+" << c << s3;
  os << c << "+";
  os << "\n";

  return os;
}

static raw_ostream &sepSpan(raw_ostream &os, char c, size_t w1, StringRef s2,
                            StringRef s3) {
  os << "| ";
  pad(os, "", w1) << " ";
  os << "+" << c << s2 << c;
  os << "+" << c << s3 << c;
  os << "+\n";

  return os;
}

void KitAttrDocEmitter::emitAttrArgs(raw_ostream &os, const Record &attr) {
  std::vector<std::string> types, names;
  const Record *type = attr.getValueAsDef("ValueType");
  if (type->isSubClassOf("BasicType")) {
    StringRef typeName = type->getValueAsString("Name");
    if (typeName == "void")
      types.push_back("");
    else
      types.push_back(quote(typeName.str(), "``"));
    names.push_back("");
  } else if (type->isSubClassOf("TupleType")) {
    std::vector<const Record *> elems = type->getValueAsListOfDefs("Elements");
    for (size_t i = 0; i < elems.size(); ++i) {
      const Record *elem = elems[i];
      const Record *elemType = elem->getValueAsDef("ElemType");
      StringRef elemName = elem->getValueAsString("ElemName");
      StringRef typeName = elemType->getValueAsString("Name");

      types.push_back(quote(typeName, "``"));
      names.push_back(quote(elemName, "``"));
    }
  }

  std::string enm = getEnum(attr);
  std::string hdr1 = "Enum";
  size_t w1 = std::max(hdr1.size(), enm.size());
  std::string dash1 = std::string(w1, '-');
  std::string eq1 = std::string(w1, '=');

  std::string hdr2 = "Value Types";
  size_t w2 = hdr2.size();
  for (StringRef type : types)
    w2 = std::max(w2, type.size());
  std::string dash2 = std::string(w2, '-');
  std::string eq2 = std::string(w2, '=');

  std::string hdr3 = "Value Names";
  size_t w3 = hdr3.size();
  for (StringRef name : names)
    w3 = std::max(w3, name.size());
  std::string dash3 = std::string(w3, '-');
  std::string eq3 = std::string(w3, '=');

  sep(os, '-', dash1, dash2, dash3);
  line(os, hdr1, w1, hdr2, w2, hdr3, w3);
  sep(os, '=', eq1, eq2, eq3);
  if (types.size() > 0) {
    line(os, enm, w1, types[0], w2, names[0], w3);
    for (size_t i = 1; i < types.size(); ++i) {
      sepSpan(os, '-', w1, dash2, dash3);
      line(os, "", w1, types[i], w2, names[i], w3);
    }
  } else {
    line(os, enm, w1, "", w2, "", w3);
  }
  sep(os, '-', dash1, dash2, dash3);
  os << "\n";
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
