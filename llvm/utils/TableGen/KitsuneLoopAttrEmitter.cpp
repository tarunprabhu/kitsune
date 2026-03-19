//===- KitsuneLoopAttrsEmitter.cpp - Generate loop attributes -------------===//
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

#define DEBUG_TYPE "kitsune-loop-attrs-emitter"

using namespace llvm;

namespace {

class LoopAttrsEmitter {
private:
  const RecordKeeper &records;
  SmallVector<StringRef, 4> attrKinds;

private:
  void emitAttrs(raw_ostream &os, StringRef kind);
  void emitAttrs(raw_ostream &os);
  void emitAttrEnums(raw_ostream &os);

public:
  LoopAttrsEmitter(const RecordKeeper &records);
  void run(raw_ostream &os);
};

} // namespace

static StringRef getBaseMacroName(bool tapirLoopsOnly) {
  if (tapirLoopsOnly)
    return "TAPIR_LOOP_ATTR";
  else
    return "LOOP_ATTR";
}

static std::string getMacroName(StringRef kind, bool tapirLoopsOnly) {
  assert(kind.ends_with("Attr") && "Kind name must end in 'Attr'");
  std::string name = kind.drop_back(4).upper();
  if (tapirLoopsOnly)
    return "TAPIR_LOOP_ATTRIBUTE_" + name;
  else
    return "LOOP_ATTRIBUTE_" + name;
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

static StringRef getIRType(StringRef kind) {
  return StringSwitch<StringRef>(kind)
      .Cases("EnumAttr", "FlagAttr", "Int32Attr", "int32_t")
      .Case("Int64Attr", "int64_t")
      .Case("StrAttr", "StringRef");
}

void LoopAttrsEmitter::emitAttrs(raw_ostream &os, StringRef kind) {
  StringRef macroArgs = getMacroArgs(kind);
  StringRef attrType = getAttrType(kind);
  StringRef irType = getIRType(kind);

  for (bool tapirLoopsOnly : {false, true}) {
    StringRef baseMacro = getBaseMacroName(tapirLoopsOnly);
    std::string macro = getMacroName(kind, tapirLoopsOnly);

    os << "#ifndef " << macro << "\n";
    os << "#define " << macro << macroArgs << " \\\n";
    os << "  ";
    os << baseMacro << "(NAME, " << attrType << ", IRNAME, " << irType << ")\n";
    os << "#endif // " << macro << "\n";
    os << "\n";
  }

  for (bool tapirLoopsOnly : {false, true}) {
    for (const Record *r : records.getAllDerivedDefinitions(kind)) {
      if (isTapirLoopOnly(*r) == tapirLoopsOnly) {
        std::string macro = getMacroName(kind, tapirLoopsOnly);
        StringRef attrName = r->getName();
        std::string irName = getLoopAttrIRName(*r);

        os << macro << "(" << attrName << ", \"" << irName << "\"";
        if (kind == "EnumAttr")
          os << ", " << r->getValueAsString("ValueType");
        os << ")\n";
      }
    }
  }

  os << "\n";
  for (bool tapirLoopsOnly : {true, false})
    os << "#undef " << getMacroName(kind, tapirLoopsOnly) << "\n";
}

void LoopAttrsEmitter::emitAttrs(raw_ostream &os) {
  os << "#ifdef GET_LOOP_ATTRS\n";
  os << "#undef GET_LOOP_ATTRS\n";
  os << "\n";

  for (bool tapirLoopsOnly : {false, true}) {
    StringRef macro = getBaseMacroName(tapirLoopsOnly);
    os << "#ifndef " << macro << "\n";
    os << "#define " << macro << "(NAME, TYPE, IRNAME, IRTYPE)\n";
    os << "#endif // " << macro << "\n";
    os << "\n";
  }
  os << "\n";

  for (StringRef kind : attrKinds) {
    emitAttrs(os, kind);
    os << "\n";
  }

  os << "\n";
  for (bool tapirLoopsOnly : {true, false})
    os << "#undef " << getBaseMacroName(tapirLoopsOnly) << "\n";
  os << "\n";
  os << "#endif // GET_LOOP_ATTRS\n";
}

void LoopAttrsEmitter::emitAttrEnums(raw_ostream &os) {
  os << "#ifdef GET_LOOP_ATTR_ENUMS\n";
  os << "#undef GET_LOOP_ATTR_ENUMS\n";
  os << "\n";

  unsigned val = 1;
  for (StringRef kind : attrKinds) {
    for (const Record *r : records.getAllDerivedDefinitions(kind)) {
      os << r->getName() << " = " << val << ",\n";
      ++val;
    }
  }

  os << "\n";
  os << "#endif // GET_LOOP_ATTR_ENUMS\n";
}

void LoopAttrsEmitter::run(raw_ostream &os) {
  emitAttrs(os);
  os << "\n\n";
  emitAttrEnums(os);
}

LoopAttrsEmitter::LoopAttrsEmitter(const RecordKeeper &records)
    : records(records) {
  for (const auto &[_, r] : records.getClasses()) {
    if (!r->isSubClassOf("Attr"))
      continue;
    StringRef name = r->getName();
    if (!name.ends_with("Attr"))
      PrintFatalError("All attribute kind names must end in 'Attr'");
    attrKinds.push_back(name);
  }
  std::sort(attrKinds.begin(), attrKinds.end());
}

static TableGen::Emitter::OptClass<LoopAttrsEmitter>
    X("gen-kitsune-loop-attrs", "Generate Kitsune-specific loop attributes");
