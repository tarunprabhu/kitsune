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
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

static StringRef toString(bool b) { return b ? "true" : "false"; }

static bool isBasic(const Record &type) {
  return type.isSubClassOf("BasicType");
}

static bool isList(const Record &type) { return type.isSubClassOf("ListType"); }

static bool isSet(const Record &type) { return type.isSubClassOf("SetType"); }

static bool isTuple(const Record &type) {
  return type.isSubClassOf("TupleType");
}

std::string KitAttrHeaderEmitter::getBaseMacroName() const {
  std::string buf;
  raw_string_ostream os(buf);

  os << getMacroRoot() << "_ATTR";
  os.flush();

  return buf;
}

StringRef KitAttrHeaderEmitter::getBaseMacroArgs() const {
  return "(NAME, IRNAME, CUSTOMVERIFY, TYPE)";
}

std::string KitAttrHeaderEmitter::getElemMacroName() const {
  return getBaseMacroName() + "_N";
}

StringRef KitAttrHeaderEmitter::getElemMacroArgs() const {
  return "(NAME, IRNAME, ETY, ENAME, EN, NELEMS)";
}

std::string KitAttrHeaderEmitter::getMacroName(unsigned n) const {
  std::string buf;
  raw_string_ostream os(buf);

  os << getBaseMacroName() << "_" << n;
  os.flush();

  return buf;
}

std::string KitAttrHeaderEmitter::getMacroName(const Record &type) const {
  std::string buf;
  raw_string_ostream os(buf);

  os << getBaseMacroName() << "_";
  if (isBasic(type))
    if (type.getValueAsString("Name") == "void")
      os << 0;
    else
      os << 1;
  else if (isList(type))
    os << "L";
  else if (isSet(type))
    os << "S";
  else if (isTuple(type))
    os << type.getValueAsListOfDefs("Elements").size();
  os.flush();

  return buf;
}

std::string KitAttrHeaderEmitter::getMacroName(StringRef kind) const {
  std::string buf;
  raw_string_ostream os(buf);

  os << getBaseMacroName() << "_" << kind;
  os.flush();

  return buf;
}

std::string KitAttrHeaderEmitter::getMacroArgs(unsigned n) const {
  std::string buf;
  raw_string_ostream os(buf);

  switch (n) {
  case 0:
    os << "(NAME, IRNAME, CUSTOMVERIFY)";
    break;
  case 1:
    os << "(NAME, IRNAME, CUSTOMVERIFY, TYPE)";
    break;
  default:
    os << "(NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0";
    for (unsigned i = 2; i <= n; ++i) {
      unsigned argNo = i - 1;
      os << ", ETY" << argNo << ", ENAME" << argNo << ", EN" << argNo;
    }
    os << ")";
  }
  os.flush();

  return buf;
}

std::string KitAttrHeaderEmitter::getMacroArgs(StringRef kind) const {
  std::string buf;
  raw_string_ostream os(buf);

  os << "(NAME, IRNAME, CUSTOMVERIFY, TYPE)";
  os.flush();

  return buf;
}

std::string KitAttrHeaderEmitter::getIRName(const Record &attr) const {
  return ::getIRName(getIRNamePrefix(attr), attr);
}

void KitAttrHeaderEmitter::emitMacroDefn(raw_ostream &os, unsigned n) {
  std::string baseMacroName = getBaseMacroName();
  std::string elemMacroName = getElemMacroName();
  std::string macroName = getMacroName(n);
  std::string macroArgs = getMacroArgs(n);

  os << "#ifndef " << macroName << "\n";
  os << "#define " << macroName << macroArgs;
  switch (n) {
  case 0:
    os << " \\\n    " << baseMacroName << "(NAME, IRNAME, CUSTOMVERIFY,)";
    break;
  case 1:
    os << " \\\n    " << baseMacroName << "(NAME, IRNAME, CUSTOMVERIFY, TYPE)";
    break;
  default:
    os << " \\\n    " << baseMacroName << "(NAME, IRNAME, CUSTOMVERIFY,)";
    for (unsigned i = 0; i < n; ++i)
      os << " \\\n    " << elemMacroName << "(NAME, IRNAME, ETY" << i
         << ", ENAME" << i << ", EN" << i << ", " << n << ")";
    break;
  }
  os << "\n";
  os << "#endif // " << macroName << "\n";
  os << "\n";
}

void KitAttrHeaderEmitter::emitMacroDefn(raw_ostream &os, StringRef kind) {
  std::string baseMacroName = getBaseMacroName();
  std::string elemMacroName = getElemMacroName();
  std::string macroName = getMacroName(kind);
  std::string macroArgs = getMacroArgs(kind);

  os << "#ifndef " << macroName << "\n";
  os << "#define " << macroName << macroArgs;
  os << " \\\n    " << baseMacroName << "(NAME, IRNAME, CUSTOMVERIFY, TYPE)";
  os << "\n";
  os << "#endif // " << macroName << "\n";
  os << "\n";
}

void KitAttrHeaderEmitter::emitAttr(raw_ostream &os, const Record &attr) {
  const Record *type = attr.getValueAsDef("ValueType");

  std::vector<std::string> args;
  if (isBasic(*type)) {
    StringRef typeName = type->getValueAsString("Name");
    if (typeName != "void")
      args.push_back(typeName.str());
  } else if (isList(*type) || isSet(*type)) {
    const Record *elemType = type->getValueAsDef("ElemType");
    StringRef elemTypeName = elemType->getValueAsString("Name");
    args.push_back(elemTypeName.str());
  } else if (isTuple(*type)) {
    std::vector<const Record *> elems = type->getValueAsListOfDefs("Elements");
    for (size_t i = 0; i < elems.size(); ++i) {
      const Record *elem = elems[i];
      const Record *elemType = elem->getValueAsDef("ElemType");
      StringRef elemName = elem->getValueAsString("ElemName");

      args.push_back(elemType->getValueAsString("Name").str());
      args.push_back(elemName.str());
      args.push_back(std::to_string(i));
    }
  } else {
    PrintFatalError(attr.getLoc(), "Attribute type is not supported");
  }

  StringRef attrName = attr.getName();
  std::string irName = quote(getIRName(attr));
  std::string macroName = getMacroName(*type);
  bool hasCustomVerifier = attr.getValueAsBit("HasCustomVerifier");

  os << macroName << "(" << attrName;
  os << ", " << irName;
  os << ", " << toString(hasCustomVerifier);
  for (StringRef arg : args)
    os << ", " << arg;
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

  std::string elemMacroName = getElemMacroName();
  os << "#ifndef " << elemMacroName << "\n";
  os << "#define " << elemMacroName << getElemMacroArgs() << "\n";
  os << "#endif // " << elemMacroName << "\n";
  os << "\n";
}

void KitAttrHeaderEmitter::emitMacroDefs(raw_ostream &os) {
  for (unsigned i = 0; i <= MaxTupleElements; ++i)
    emitMacroDefn(os, i);
  emitMacroDefn(os, "L");
  emitMacroDefn(os, "S");
}

void KitAttrHeaderEmitter::emitAttrs(raw_ostream &os) {
  for (const Record *attr : records.getAllDerivedDefinitions(getAttrBase()))
    emitAttr(os, *attr);
  os << "\n";
}

void KitAttrHeaderEmitter::emitMacroUndefs(raw_ostream &os) {
  os << "#undef " << getMacroName("S") << "\n";
  os << "#undef " << getMacroName("L") << "\n";
  for (unsigned i = MaxTupleElements + 1; i > 0; --i)
    os << "#undef " << getMacroName(i - 1) << "\n";
}

void KitAttrHeaderEmitter::emitBaseMacroUndef(raw_ostream &os) {
  os << "#undef " << getElemMacroName() << "\n";
  os << "#undef " << getBaseMacroName() << "\n";
}

void KitAttrHeaderEmitter::emitAttrsGuardOut(raw_ostream &os) {
  os << "\n" << "#endif // GET_" << getMacroRoot() << "_ATTRS\n";
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
  auto checkBasicType = [](const Record &type, const Record &attr) -> void {
    StringRef typeName = type.getValueAsString("Name");
    if (typeName.ends_with("*") && typeName != "llvm::MDNode*")
      PrintFatalError(attr.getLoc(),
                      "Only pointers to raw MDNode's are supported");
  };

  for (const Record *attr : records.getAllDerivedDefinitions(getAttrBase())) {
    const Record *type = attr->getValueAsDef("ValueType");
    if (isTuple(*type)) {
      size_t n = type->getValueAsListOfDefs("Elements").size();
      if (n < MinTupleElements)
        PrintFatalError(attr->getLoc(), "Not enough elements in tuple");
      else if (n > MaxTupleElements)
        PrintFatalError(attr->getLoc(), "Too many elements in tuple");
    } else if (isBasic(*type)) {
      checkBasicType(*type, *attr);
    } else if (isList(*type) || isSet(*type)) {
      const Record *elemType = type->getValueAsDef("ElemType");
      if (!isBasic(*elemType))
        PrintFatalError(attr->getLoc(),
                        "Element of attribute not a basic type");
      checkBasicType(*elemType, *attr);
      StringRef typeName = elemType->getValueAsString("Name");
      if (typeName == "void")
        PrintFatalError(attr->getLoc(), "Element type cannot be void");
    } else {
      PrintFatalError(attr->getLoc(),
                      "Type of value not a basic or tuple type");
    }
  }

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
    : records(records) {}
