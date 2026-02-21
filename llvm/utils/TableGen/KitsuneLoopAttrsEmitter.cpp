//===- KitsuneLoopAttrsEmitter.cpp - Generate loop attributes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kitsune-loop-attrs-emitter"

using namespace llvm;

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

static std::string getMacroName(StringRef kind, bool tapirLoopsOnly) {
  std::string buf;
  raw_string_ostream os(buf);

  if (tapirLoopsOnly)
    os << "TAPIR_";
  // This assumes that the attribute names always end in Attr.
  os << "LOOP_ATTRIBUTE_" << kind.drop_back(4).upper();

  os.flush();
  return buf;
}

static StringRef getMacroArgs(StringRef kind) {
  if (kind == "EnumAttr")
    return "(NAME, IRNAME, TYPE)";
  return "(NAME, IRNAME)";
}

static StringRef getAttrType(StringRef kind) {
  return StringSwitch<StringRef>(kind)
      .Case("EnumAttr", "TYPE")
      .Case("FlagAttr", "")
      .Case("Int32Attr", "int32_t")
      .Case("Int64Attr", "int64_t")
      .Case("StrAttr", "StringRef");
}

static std::string getIRName(StringRef attrName, bool tapirLoopsOnly) {
  auto addDot = [](char c, char prev) -> bool {
    return (std::isalpha(prev) && std::isdigit(c)) ||
           (std::isdigit(prev) && std::isalpha(c)) ||
           (std::islower(prev) && std::isupper(c));
  };

  std::string buf;
  raw_string_ostream os(buf);

  if (tapirLoopsOnly)
    os << "tapir.";
  os << "loop.";
  os << (char)std::tolower(attrName[0]);
  for (unsigned i = 1, ie = attrName.size(); i < ie; ++i) {
    if (addDot(attrName[i], attrName[i - 1]))
      os << ".";
    os << (char)std::tolower(attrName[i]);
  }

  os.flush();
  return buf;
}

static StringRef getIRType(StringRef kind) {
  return StringSwitch<StringRef>(kind)
      .Cases("EnumAttr", "FlagAttr", "Int32Attr", "int32_t")
      .Case("Int64Attr", "int64_t")
      .Case("StrAttr", "StringRef");
}

class TapirLoopAttrsEmitter {
private:
  const RecordKeeper &records;

  // IMPORTANT: All kinds here must end in Attr.
  SmallVector<StringRef, 4> attrKinds = {
      "EnumAttr", "FlagAttr", "Int32Attr", "Int64Attr", "StrAttr",
  };

private:
  void emitAttrs(raw_ostream &os, StringRef kind) {
    StringRef macroArgs = getMacroArgs(kind);
    StringRef attrType = getAttrType(kind);
    StringRef irType = getIRType(kind);

    for (bool tapirLoopsOnly : {false, true}) {
      std::string macro = getMacroName(kind, tapirLoopsOnly);

      line(os, {"//", kind});
      line(os, {"#ifndef", macro});
      os << "#define " << macro << macroArgs << " \\\n";
      os << "  ";
      if (tapirLoopsOnly)
        os << "TAPIR_";
      os << "LOOP_ATTR(NAME, " << attrType << ", IRNAME, " << irType << ")\n";
      line(os, {"#endif //", macro});
      line(os);
    }

    for (const Record *r : records.getAllDerivedDefinitions(kind)) {
      const Record *allowedOn = r->getValueAsDef("AllowedOn");
      bool tapirLoopsOnly = allowedOn->getName() == "TapirLoopsOnly";
      std::string macro = getMacroName(kind, tapirLoopsOnly);
      StringRef attrName = r->getName();
      std::string irName = getIRName(attrName, tapirLoopsOnly);

      os << macro << "(" << attrName << ", \"" << irName << "\"";
      if (kind == "EnumAttr")
        os << ", " << r->getValueAsString("ValueType");
      os << ")\n";
    }

    line(os);
    line(os, {"#undef", getMacroName(kind, /*tapirLoopsOnly=*/true)});
    line(os, {"#undef", getMacroName(kind, /*tapirLoopsOnly=*/false)});
  }

  void emitAttrs(raw_ostream &os) {
    line(os, "#ifdef GET_LOOP_ATTRS");
    line(os, "#undef GET_LOOP_ATTRS");
    line(os);
    line(os, "#ifndef LOOP_ATTR");
    line(os, "#define LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)");
    line(os, "#endif // LOOP_ATTR");
    line(os);
    line(os, "#ifndef TAPIR_LOOP_ATTR");
    line(os, "#define TAPIR_LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)");
    line(os, "#endif // TAPIR_LOOP_ATTR");
    line(os);

    for (StringRef kind : attrKinds) {
      emitAttrs(os, kind);
      line(os);
    }

    line(os, "#undef TAPIR_LOOP_ATTR");
    line(os, "#undef LOOP_ATTR");
    line(os, "#endif // GET_LOOP_ATTRS");
  }

  void emitAttrEnums(raw_ostream &os) {
    line(os, "#ifdef GET_LOOP_ATTR_ENUMS");
    line(os, "#undef GET_LOOP_ATTR_ENUMS");
    line(os);

    unsigned val = 1;
    for (StringRef kind : attrKinds) {
      for (const Record *r : records.getAllDerivedDefinitions(kind)) {
        os << r->getName() << " = " << val << ",\n";
        ++val;
      }
    }

    line(os);
    line(os, "#endif // GET_LOOP_ATTR_ENUMS");
  }

public:
  TapirLoopAttrsEmitter(const RecordKeeper &records) : records(records) {}

  void run(raw_ostream &os) {
    emitAttrs(os);
    line(os);
    line(os, "// ------------------------------------------------------------");
    line(os);
    emitAttrEnums(os);
  }
};

static TableGen::Emitter::OptClass<TapirLoopAttrsEmitter>
    X("gen-kitsune-loop-attrs", "Generate Kitsune-specific loop attributes");
