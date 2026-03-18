//===- KitsuneInstAttrEmitter.cpp - Generate instruction attributes -------===//
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

#define DEBUG_TYPE "kitsune-inst-attrs-emitter"

using namespace llvm;

static std::string getMacroName(StringRef kind) {
  assert(kind.ends_with("Attr") && "Attribute kind must end in 'Attr'");
  return "INST_ATTRIBUTE_" + kind.drop_back(4).upper();
}

static StringRef getMacroArgs(StringRef kind) {
  if (kind == "EnumAttr")
    return "(NAME, IRNAME, TYPE)";
  return "(NAME, IRNAME)";
}

static StringRef getValueType(StringRef kind) {
  return StringSwitch<StringRef>(kind)
      .Case("EnumAttr", "TYPE")
      .Case("FlagAttr", "")
      .Case("Int32Attr", "int32_t")
      .Case("Int64Attr", "int64_t")
      .Case("LoopAttr", "Loop*")
      .Case("MDNodeAttr", "MDNode*")
      .Case("StrAttr", "StringRef");
}

static std::string getIRName(const Record &r) {
  return "kit.inst." + getBaseName(r);
}

static StringRef getIRType(StringRef kind) {
  return StringSwitch<StringRef>(kind)
      .Cases("EnumAttr", "FlagAttr", "Int32Attr", "int32_t")
      .Case("Int64Attr", "int64_t")
      .Case("LoopAttr", "Loop*")
      .Case("MDNodeAttr", "MDNode*")
      .Case("StrAttr", "StringRef");
}

class InstAttrsEmitter {
private:
  const RecordKeeper &records;
  SmallVector<StringRef, 8> attrKinds;

private:
  raw_ostream &emitAttrs(raw_ostream &os, StringRef kind) {
    std::string macroName = getMacroName(kind);
    StringRef macroArgs = getMacroArgs(kind);
    StringRef valType = getValueType(kind);
    StringRef irType = getIRType(kind);

    os << "#ifndef " << macroName << "\n";
    os << "#define " << macroName << macroArgs << " \\\n";
    os << "  ";
    os << "INST_ATTR(NAME, " << valType << ", IRNAME, " << irType << ")\n";
    os << "#endif // " << macroName << "\n";

    for (const Record *r : records.getAllDerivedDefinitions(kind)) {
      std::string macro = getMacroName(kind);
      StringRef attrName = r->getName();
      std::string irName = getIRName(*r);

      os << macro << "(" << attrName << ", \"" << irName << "\"";
      if (kind == "EnumAttr")
        os << ", " << r->getValueAsString("ValueType");
      os << ")\n";
    }

    os << "#undef " << macroName << "\n";

    return os;
  }

  raw_ostream &emitAttrs(raw_ostream &os) {
    os << "#ifdef GET_INST_ATTRS" << "\n";
    os << "#undef GET_INST_ATTRS" << "\n";
    os << "\n";
    os << "#ifndef INST_ATTR" << "\n";
    os << "#define INST_ATTR(NAME, TYPE, IRNAME, IRTYPE)" << "\n";
    os << "#endif // INST_ATTR" << "\n";
    os << "\n";

    for (StringRef kind : attrKinds) {
      emitAttrs(os, kind);
      os << "\n";
    }

    os << "#undef INST_ATTR" << "\n";
    os << "#endif // GET_INST_ATTRS" << "\n";

    return os;
  }

  raw_ostream &emitAttrEnums(raw_ostream &os) {
    os << "#ifdef GET_INST_ATTR_ENUMS" << "\n";
    os << "#undef GET_INST_ATTR_ENUMS" << "\n";
    os << "\n";

    unsigned val = 1;
    for (const Record *r : records.getAllDerivedDefinitions("Attr")) {
      os << r->getName() << " = " << val << ",\n";
      ++val;
    }

    os << "\n";
    os << "#endif // GET_INST_ATTR_ENUMS" << "\n";

    return os;
  }

public:
  InstAttrsEmitter(const RecordKeeper &records) : records(records) {
    for (const auto &[_, r] : records.getClasses()) {
      StringRef name = r->getName();
      if (name == "Attr")
        continue;
      if (!r->isSubClassOf("Attr"))
        PrintFatalError("All classes must be subclasses of Attr");
      if (!name.ends_with("Attr"))
        PrintFatalError("All attribute kind names must end in 'Attr'");
      attrKinds.push_back(name);
    }
    std::sort(attrKinds.begin(), attrKinds.end());
  }

  void run(raw_ostream &os) {
    emitAttrs(os);
    os << "\n\n";
    emitAttrEnums(os);
  }
};

static TableGen::Emitter::OptClass<InstAttrsEmitter>
    X("gen-kitsune-inst-attrs", "Generate Kitsune-specific loop attributes");
