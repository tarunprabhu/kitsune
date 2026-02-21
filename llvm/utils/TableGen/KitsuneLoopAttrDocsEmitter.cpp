//===- KitsuneLoopAttrDocsEmitter.cpp - Generate docs for loop attributes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KitsuneLoopAttrUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kitsune-loop-attr-docs-emitter"

using namespace llvm;

static std::string getSectionLabel(const Record &attr) {
  std::string attrName = getBaseName(attr);
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

static StringRef getAllowedOn(const Record &attr) {
  const Record *allowedOn = attr.getValueAsDef("AllowedOn");
  return StringSwitch<StringRef>(allowedOn->getName())
      .Case("TapirLoopsOnly", "Tapir loops only")
      .Case("NormalLoopsOnly", "Normal loops only")
      .Default("All loops");
}

class LoopAttrDocsEmitter {
private:
  const RecordKeeper &records;

private:
  void emitAttr(const Record &attr, raw_ostream &os) {
    std::string irName = getIRName(attr);
    os << ".. _loop-attr-" << getSectionLabel(attr) << ":\n\n";
    os << irName << "\n";
    os << std::string(irName.size(), '-') << "\n";
    os << "\n";
    os << ".. csv-table::\n";
    os << "  :header: " << quote("Enum") << ", " << quote("Value Type") << ", "
       << quote("Allowed On") << "\n";
    os << "\n";
    os << "  " << quote(getAttrName(attr)) << ", " << quote(getValueType(attr))
       << ", " << quote(getAllowedOn(attr)) << "\n";
    os << "\n";

    os << attr.getValueAsString("Documentation") << "\n";
  }

public:
  LoopAttrDocsEmitter(const RecordKeeper &records) : records(records) {}

  void run(raw_ostream &os) {
    std::vector<const Record *> attrRecords;
    for (const Record *r : records.getAllDerivedDefinitions("Attr"))
      attrRecords.push_back(r);
    std::sort(attrRecords.begin(), attrRecords.end(),
              [](const Record *l, const Record *r) -> bool {
                return l->getName() < r->getName();
              });

    const Record *globalDoc = records.getDef("GlobalDocumentation");
    if (!globalDoc) {
      PrintFatalError(
          "The GlobalDocumentation top-level definition is missing, "
          "no documentation will be generated.");
      return;
    }

    os << globalDoc->getValueAsString("Intro");
    for (const Record *r : attrRecords) {
      os << "\n";
      emitAttr(*r, os);
    }
  }
};

static TableGen::Emitter::OptClass<LoopAttrDocsEmitter>
    X("gen-kitsune-loop-attr-docs",
      "Generate documentation for Kitsune-specific loop attributes");
