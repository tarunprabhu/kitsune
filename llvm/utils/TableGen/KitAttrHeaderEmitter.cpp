//===- KitAttrHeaderEmitter.cpp - Base class to emit attribute headers ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for emitters that generate headers for Kitsune-specific attributes
//
//===----------------------------------------------------------------------===//

#include "KitAttrHeaderEmitter.h"
#include "KitAttrCommon.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

KitAttrHeaderEmitter::Kind KitAttrHeaderEmitter::getKind(const Record &type) {
  StringRef typeName = type.getValueAsString("Name");

  if (type.getValueAsDef("IsEnum")->getName() == "True") {
    return {"ENUM", "TYPE"};
  } else if (typeName == "void") {
    return {"FLAG", ""};
  } else if (typeName.ends_with("*")) {
    StringRef kind = typeName.drop_back(1);
    if (kind.starts_with("llvm::"))
      kind = kind.drop_front(6);
    return {kind.upper(), typeName};
  } else {
    return {StringSwitch<std::string>(typeName)
                .Case("llvm::StringRef", "STR")
                .Case("int32_t", "I32")
                .Case("int64_t", "I64")
                .Case("float", "F32")
                .Case("double", "F64")
                .Case("uint32_t", "U32")
                .Case("uint64_t", "U64"),
            typeName};
  }
}

StringRef KitAttrHeaderEmitter::getAttrBase() const { return "AttrBase"; }

std::string KitAttrHeaderEmitter::getBaseMacroName() const {
  std::string buf;
  raw_string_ostream os(buf);

  os << getMacroRoot() << "_ATTR";
  os.flush();

  return buf;
}

StringRef KitAttrHeaderEmitter::getBaseMacroArgs() const {
  return "(NAME, TYPE, IRNAME)";
}

std::string KitAttrHeaderEmitter::getMacroName(const Kind &kind) const {
  std::string buf;
  raw_string_ostream os(buf);

  os << getBaseMacroName() << "_" << kind.name;
  os.flush();

  return buf;
}

StringRef KitAttrHeaderEmitter::getMacroArgs(const Kind &kind) const {
  if (kind.name == "ENUM")
    return "(NAME, IRNAME, TYPE)";
  return "(NAME, IRNAME)";
}

std::string KitAttrHeaderEmitter::getIRName(const Record &attr) const {
  return ::getIRName(getIRNamePrefix(attr), attr);
}

void KitAttrHeaderEmitter::emitMacroDefn(raw_ostream &os, const Kind &kind) {
  StringRef baseMacroName = getBaseMacroName();
  std::string macroName = getMacroName(kind);
  StringRef macroArgs = getMacroArgs(kind);

  os << "#ifndef " << macroName << "\n";
  os << "#define " << macroName << macroArgs << " ";
  os << baseMacroName << "(NAME, " << kind.type << ", IRNAME)\n";
  os << "#endif // " << macroName << "\n";
  os << "\n";
}

void KitAttrHeaderEmitter::emitAttr(raw_ostream &os, const Record &attr) {
  const Record *type = attr.getValueAsDef("ValueType");
  Kind kind = getKind(*type);
  std::string macroName = getMacroName(kind);
  std::string irName = getIRName(attr);
  StringRef attrName = attr.getName();

  os << macroName << "(" << attrName << ", \"" << irName << "\"";
  if (kind.name == "ENUM")
    os << ", " << type->getValueAsString("Name");
  os << ")\n";
}

void KitAttrHeaderEmitter::emitAttrsGuardIn(raw_ostream &os) {
  StringRef root = getMacroRoot();

  os << "#ifdef GET_" << root << "_ATTRS\n";
  os << "#define GET_" << root << "_ATTRS\n";
  os << "\n";
}

void KitAttrHeaderEmitter::emitBaseMacroDef(raw_ostream &os) {
  std::string baseMacroName = getBaseMacroName();

  os << "#ifndef " << baseMacroName << "\n";
  os << "#define " << baseMacroName << getBaseMacroArgs() << "\n";
  os << "#endif // " << baseMacroName << "\n";
  os << "\n";
}

void KitAttrHeaderEmitter::emitMacroDefs(raw_ostream &os) {
  for (const Kind &kind : attrKinds)
    emitMacroDefn(os, kind);
}

void KitAttrHeaderEmitter::emitAttrs(raw_ostream &os) {
  for (const Record *attr : records.getAllDerivedDefinitions(getAttrBase()))
    emitAttr(os, *attr);
  os << "\n";
}

void KitAttrHeaderEmitter::emitMacroUndefs(raw_ostream &os) {
  for (const Kind &kind : attrKinds)
    os << "#undef " << getMacroName(kind) << "\n";
  os << "\n";
}

void KitAttrHeaderEmitter::emitBaseMacroUndef(raw_ostream &os) {
  os << "#undef " << getBaseMacroName() << "\n";
  os << "\n";
}

void KitAttrHeaderEmitter::emitAttrsGuardOut(raw_ostream &os) {
  os << "#endif // GET_" << getMacroRoot() << "_ATTRS\n";
}

void KitAttrHeaderEmitter::emitEnums(raw_ostream &os) {
  StringRef root = getMacroRoot();

  os << "#ifdef GET_" << root << "_ATTR_ENUMS\n";
  os << "#undef GET_" << root << "_ATTR_ENUMS\n";
  os << "\n";

  unsigned val = 1;
  for (const Record *attr : records.getAllDerivedDefinitions(getAttrBase())) {
    os << attr->getName() << " = " << val << ",\n";
    ++val;
  }

  os << "\n";
  os << "#endif // GET_" << root << "_ATTR_ENUMS\n";
}

void KitAttrHeaderEmitter::run(raw_ostream &os) {
  emitAttrsGuardIn(os);
  emitBaseMacroDef(os);
  emitMacroDefs(os);
  emitAttrs(os);
  emitMacroUndefs(os);
  emitBaseMacroUndef(os);
  emitAttrsGuardOut(os);

  os << "\n\n";

  emitEnums(os);
}

KitAttrHeaderEmitter::KitAttrHeaderEmitter(const RecordKeeper &records)
    : records(records) {
  // Get all the types for which may kinds have to be created. This will include
  // any "inline" types that may have been created.
  SmallSet<Kind, 8> kinds;
  for (const Record *type : records.getAllDerivedDefinitions("Type"))
    kinds.insert(getKind(*type));

  attrKinds.assign(kinds.begin(), kinds.end());
  std::sort(attrKinds.begin(), attrKinds.end(),
            [](const Kind &l, const Kind &r) { return l.name < r.name; });
}
