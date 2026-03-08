//===- KitsuneModuleAttrsEmitter.cpp - Generate module attributes ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KitsuneAttrUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kitsune-module-attrs-emitter"

using namespace llvm;

// The maximum number of values allowed in an attribute.
constexpr size_t MAXVALS = 8;

static raw_ostream &line(raw_ostream &os, ArrayRef<StringRef> strs = {}) {
  if (strs.size()) {
    os << strs[0];
    for (unsigned i = 1, ie = strs.size(); i < ie; ++i)
      os << " " << strs[i];
  }
  os << "\n";
  return os;
}

static raw_ostream &line(raw_ostream &os, StringRef s) {
  os << s << "\n";
  return os;
}

static std::string getIRName(const Record &attr) {
  return "kit.module." + getBaseName(attr);
}

static std::string getMacroName(const Record &attr) {
  SmallString<16> buf;
  raw_svector_ostream os(buf);

  size_t vals = attr.getValueAsListOfDefs("Values").size();
  if (vals > MAXVALS)
    PrintFatalError(attr.getLoc(),
                    "Maximum allowed values exceeded in attribute");

  os << "MODULE_ATTR_" << vals;
  return buf.c_str();
}

class ModuleAttrsEmitter {
private:
  const RecordKeeper &recordKeeper;

private:
  void emitAttrs(raw_ostream &os) {
    line(os, "#ifdef GET_MODULE_ATTRS");
    line(os, "#undef GET_MODULE_ATTRS");
    line(os);
    line(os, "#ifndef MODULE_ATTR");
    line(os, "#define MODULE_ATTR(NAME, IRNAME)");
    line(os, "#endif // MODULE_ATTR");
    line(os);

    for (size_t i = 0; i <= MAXVALS; ++i) {
      os << "#ifndef MODULE_ATTR_" << i << "\n";
      os << "#define MODULE_ATTR_" << i << "(NAME, IRNAME";
      for (size_t j = 1; j <= i; ++j)
        os << ", TY" << j << ", V" << j;
      os << ") \\\n";
      os << "    MODULE_ATTR(NAME, IRNAME)\n";
      os << "#endif // MODULE_ATTR_" << i;
      line(os);
      line(os);
    }

    for (const Record *attr : recordKeeper.getAllDerivedDefinitions("Attr")) {
      os << getMacroName(*attr) << "(";
      os << attr->getName();
      os << ", \"" << getIRName(*attr) << "\"";
      for (const Record *v : attr->getValueAsListOfDefs("Values")) {
        os << ", " << v->getValueAsDef("Type")->getValueAsString("Name");
        os << ", " << v->getValueAsString("Name");
      }
      os << ")";
      line(os);
    }
    line(os);

    for (size_t i = 0; i <= MAXVALS; ++i)
      os << "#undef MODULE_ATTR_" << (MAXVALS - i) << "\n";
    line(os, "#undef MODULE_ATTR");
    line(os);
    line(os, "#endif // GET_MODULE_ATTRS");
  }

  void emitAttrEnums(raw_ostream &os) {
    line(os, "#ifdef GET_MODULE_ATTR_ENUMS");
    line(os, "#undef GET_MODULE_ATTR_ENUMS");
    line(os);

    unsigned val = 1;
    for (const Record *r : recordKeeper.getAllDerivedDefinitions("Attr")) {
      os << r->getName() << " = " << val << ",\n";
      ++val;
    }

    line(os);
    line(os, "#endif // GET_MODULE_ATTR_ENUMS");
  }

public:
  ModuleAttrsEmitter(const RecordKeeper &recordKeeper)
      : recordKeeper(recordKeeper) {}

  void run(raw_ostream &os) {
    emitAttrs(os);
    line(os);
    emitAttrEnums(os);
  }
};

static TableGen::Emitter::OptClass<ModuleAttrsEmitter>
    X("gen-kitsune-module-attrs",
      "Generate Kitsune-specific module attributes");
