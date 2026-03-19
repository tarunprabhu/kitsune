//===- KitsuneModuleAttrsEmitter.cpp - Generate module attributes ---------===//
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

#define DEBUG_TYPE "kitsune-module-attrs-emitter"

using namespace llvm;

namespace {

class ModuleAttrsEmitter {
private:
  const RecordKeeper &recordKeeper;

private:
  void emitAttrs(raw_ostream &os);
  void emitAttrEnums(raw_ostream &os);

public:
  ModuleAttrsEmitter(const RecordKeeper &recordKeeper);

  void run(raw_ostream &os);
};

} // namespace

// The maximum number of values allowed in an attribute.
static constexpr size_t MAXVALS = 8;

static std::string getMacroName(const Record &attr) {
  std::string buf;
  raw_string_ostream os(buf);

  size_t vals = attr.getValueAsListOfDefs("Values").size();
  if (vals > MAXVALS)
    PrintFatalError(attr.getLoc(),
                    "Maximum allowed values exceeded in attribute");
  os << "MODULE_ATTR_" << vals;
  os.flush();

  return buf;
}

void ModuleAttrsEmitter::emitAttrs(raw_ostream &os) {
  os << "#ifdef GET_MODULE_ATTRS\n";
  os << "#undef GET_MODULE_ATTRS\n";
  os << "\n";
  os << "#ifndef MODULE_ATTR\n";
  os << "#define MODULE_ATTR(NAME, IRNAME)\n";
  os << "#endif // MODULE_ATTR\n";
  os << "\n";

  for (size_t i = 0; i <= MAXVALS; ++i) {
    os << "#ifndef MODULE_ATTR_" << i << "\n";
    os << "#define MODULE_ATTR_" << i << "(NAME, IRNAME";
    for (size_t j = 1; j <= i; ++j)
      os << ", TY" << j << ", V" << j;
    os << ") \\\n";
    os << "    MODULE_ATTR(NAME, IRNAME)\n";
    os << "#endif // MODULE_ATTR_" << i << "\n";
    os << "\n";
  }

  for (const Record *attr : recordKeeper.getAllDerivedDefinitions("Attr")) {
    os << getMacroName(*attr) << "(";
    os << attr->getName();
    os << ", \"" << getModuleAttrIRName(*attr) << "\"";
    for (const Record *v : attr->getValueAsListOfDefs("Values")) {
      os << ", " << v->getValueAsDef("Type")->getValueAsString("Name");
      os << ", " << v->getValueAsString("Name");
    }
    os << ")";
    os << "\n";
  }
  os << "\n";

  for (size_t i = 0; i <= MAXVALS; ++i)
    os << "#undef MODULE_ATTR_" << (MAXVALS - i) << "\n";
  os << "#undef MODULE_ATTR\n";
  os << "\n";
  os << "#endif // GET_MODULE_ATTRS\n";
}

void ModuleAttrsEmitter::emitAttrEnums(raw_ostream &os) {
  os << "#ifdef GET_MODULE_ATTR_ENUMS\n";
  os << "#undef GET_MODULE_ATTR_ENUMS\n";
  os << "\n";

  unsigned val = 1;
  for (const Record *r : recordKeeper.getAllDerivedDefinitions("Attr")) {
    os << r->getName() << " = " << val << ",\n";
    ++val;
  }

  os << "\n";
  os << "#endif // GET_MODULE_ATTR_ENUMS\n";
}

void ModuleAttrsEmitter::run(raw_ostream &os) {
  emitAttrs(os);
  os << "\n";
  emitAttrEnums(os);
}

ModuleAttrsEmitter::ModuleAttrsEmitter(const RecordKeeper &recordKeeper)
    : recordKeeper(recordKeeper) {}

static TableGen::Emitter::OptClass<ModuleAttrsEmitter>
    X("gen-kitsune-module-attrs",
      "Generate Kitsune-specific module attributes");
