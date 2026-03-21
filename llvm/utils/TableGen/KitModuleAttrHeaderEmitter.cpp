//===- KitModuleAttrHeaderEmitter.cpp - Generate header for module attrs --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KitAttrCommon.h"
#include "KitAttrHeaderEmitter.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kit-module-attr-header"

using namespace llvm;

namespace {

class KitModuleAttrEmitter : public KitAttrHeaderEmitter {
private:
  // The maximum number of values allowed in an attribute. This is a limitation
  // only because it is not clear if there is any benefit to writing a more
  // general implementation.
  static constexpr size_t MAXVALS = 8;

private:
  std::string getMacroName(const Record &attr);

protected:
  StringRef getMacroRoot() const override;
  StringRef getIRNamePrefix(const Record &attr) const override;
  StringRef getAttrBase() const override;
  StringRef getBaseMacroArgs() const override;

  void emitMacroDefs(raw_ostream &os) override;
  void emitAttr(raw_ostream &os, const Record &attr) override;
  void emitMacroUndefs(raw_ostream &os) override;

public:
  KitModuleAttrEmitter(const RecordKeeper &recordKeeper);
};

} // namespace

std::string KitModuleAttrEmitter::getMacroName(const Record &attr) {
  std::string buf;
  raw_string_ostream os(buf);
  StringRef base = getMacroRoot();

  os << base << "_ATTR_" << attr.getValueAsListOfDefs("Values").size();
  os.flush();

  return buf;
}

StringRef KitModuleAttrEmitter::getMacroRoot() const { return "MODULE"; }

StringRef KitModuleAttrEmitter::getIRNamePrefix(const Record &attr) const {
  return "kit.module.";
}

StringRef KitModuleAttrEmitter::getAttrBase() const { return "ModuleAttr"; }

StringRef KitModuleAttrEmitter::getBaseMacroArgs() const {
  return "(NAME, IRNAME)";
}

void KitModuleAttrEmitter::emitMacroDefs(raw_ostream &os) {
  std::string baseMacroName = getBaseMacroName();
  for (size_t i = 0; i <= MAXVALS; ++i) {
    os << "#ifndef " << baseMacroName << "_" << i << "\n";
    os << "#define " << baseMacroName << "_" << i << "(NAME, IRNAME";
    for (size_t j = 1; j <= i; ++j)
      os << ", TY" << j << ", V" << j;
    os << ") \\\n";
    os << "    " << baseMacroName << getBaseMacroArgs() << "\n";
    os << "#endif // " << baseMacroName << "_" << i << "\n";
    os << "\n";
  }
}

void KitModuleAttrEmitter::emitAttr(raw_ostream &os, const Record &attr) {
  os << getMacroName(attr) << "(";
  os << attr.getName();
  os << ", \"" << getIRName(attr) << "\"";
  for (const Record *v : attr.getValueAsListOfDefs("Values")) {
    os << ", " << getTypeName(*v);
    os << ", " << v->getValueAsString("ValueName");
  }
  os << ")";
  os << "\n";
}

void KitModuleAttrEmitter::emitMacroUndefs(raw_ostream &os) {
  for (size_t i = 0; i <= MAXVALS; ++i)
    os << "#undef " << getBaseMacroName() << "_" << (MAXVALS - i) << "\n";
  os << "\n";
}

KitModuleAttrEmitter::KitModuleAttrEmitter(const RecordKeeper &records)
    : KitAttrHeaderEmitter(records) {
  for (const Record *attr : records.getAllDerivedDefinitions(getAttrBase())) {
    size_t vals = attr->getValueAsListOfDefs("Values").size();
    if (vals > MAXVALS)
      PrintFatalError(attr->getLoc(),
                      "Maximum allowed values exceeded in attribute");
  }
}

static TableGen::Emitter::OptClass<KitModuleAttrEmitter>
    X("gen-kit-module-attr-header",
      "Generate header for Kitsune-specific module attributes");
