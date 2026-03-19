//===- KitsuneInstAttrDocsEmitter.cpp - Docs for instruction attributes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KitsuneAttrUtils.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kitsune-inst-attr-docs-emitter"

using namespace llvm;

namespace {

class InstAttrDocsEmitter {
private:
  const RecordKeeper &records;

private:
  void emitAttr(const Record &attr, raw_ostream &os);

public:
  InstAttrDocsEmitter(const RecordKeeper &records);
  void run(raw_ostream &os);
};

} // namespace

static std::string getSectionLabel(const Record &attr) {
  std::string attrName = getAttrBaseName(attr);
  std::replace(attrName.begin(), attrName.end(), '.', '-');

  return attrName;
}

static std::string quote(StringRef s) {
  std::string buf;
  raw_string_ostream os(buf);

  os << "\"" << s << "\"";
  os.flush();
  return buf;
}

static std::string getAttrName(const Record &attr) {
  std::string buf;
  raw_string_ostream os(buf);

  os << "``" << attr.getName() << "``";
  os.flush();
  return buf;
}

static std::string getValueType(const Record &attr) {
  std::string buf;
  raw_string_ostream os(buf);
  StringRef valueType = attr.getValueAsString("ValueType");

  if (valueType.size())
    os << "``" << valueType << "``";
  os.flush();
  return buf;
}

void InstAttrDocsEmitter::emitAttr(const Record &attr, raw_ostream &os) {
  std::string irName = getInstAttrIRName(attr);
  os << ".. _inst-attr-" << getSectionLabel(attr) << ":\n\n";
  os << irName << "\n";
  os << std::string(irName.size(), '-') << "\n";
  os << "\n";
  os << ".. csv-table::\n";
  os << "  :header: " << quote("Enum") << ", " << quote("Value Type") << "\n";
  os << "\n";
  os << "  " << quote(getAttrName(attr)) << ", " << quote(getValueType(attr))
     << "\n";
  os << "\n";

  os << attr.getValueAsString("Documentation") << "\n";
}

void InstAttrDocsEmitter::run(raw_ostream &os) {
  std::vector<const Record *> attrRecords;
  for (const Record *r : records.getAllDerivedDefinitions("Attr"))
    attrRecords.push_back(r);
  std::sort(attrRecords.begin(), attrRecords.end(),
            [](const Record *l, const Record *r) -> bool {
              return l->getName() < r->getName();
            });

  const Record *globalDoc = records.getDef("GlobalDocumentation");
  if (!globalDoc) {
    PrintFatalError("The GlobalDocumentation top-level definition is missing, "
                    "no documentation will be generated.");
    return;
  }

  os << globalDoc->getValueAsString("Intro");
  for (const Record *r : attrRecords) {
    os << "\n";
    emitAttr(*r, os);
  }
}

InstAttrDocsEmitter::InstAttrDocsEmitter(const RecordKeeper &records)
    : records(records) {}

static TableGen::Emitter::OptClass<InstAttrDocsEmitter>
    X("gen-kitsune-inst-attr-docs",
      "Generate documentation for Kitsune-specific instruction attributes");
