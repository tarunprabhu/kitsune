//===- KitLoopAttrHeaderEmitter.cpp - Generate header for loop attributes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KitAttrCommon.h"
#include "KitAttrHeaderEmitter.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kit-loop-attr-header"

using namespace llvm;

namespace {

class KitLoopAttrEmitter : public KitAttrHeaderEmitter {
protected:
  StringRef getMacroRoot() const override;
  StringRef getIRNamePrefix(const Record &attr) const override;

  StringRef getAttrBase() const override;
  StringRef getBaseMacroArgs() const override;
  StringRef getMacroArgs(const Kind &kind) const override;

  void emitMacroDefn(raw_ostream &os, const Kind &kind) override;
  void emitAttr(raw_ostream &os, const Record &attr) override;

public:
  KitLoopAttrEmitter(const RecordKeeper &records);
};

} // namespace

StringRef KitLoopAttrEmitter::getMacroRoot() const { return "LOOP"; }

StringRef KitLoopAttrEmitter::getIRNamePrefix(const Record &attr) const {
  if (isTapirOnly(attr))
    return "tapir.loop.";
  return "loop.";
}

StringRef KitLoopAttrEmitter::getAttrBase() const { return "LoopAttr"; }

StringRef KitLoopAttrEmitter::getBaseMacroArgs() const {
  return "(NAME, TYPE, TAPIRONLY, IRNAME)";
}

StringRef KitLoopAttrEmitter::getMacroArgs(const Kind &kind) const {
  if (kind.name == "ENUM")
    return "(NAME, IRNAME, TAPIRONLY, TYPE)";
  return "(NAME, IRNAME, TAPIRONLY)";
}

void KitLoopAttrEmitter::emitMacroDefn(raw_ostream &os, const Kind &kind) {
  std::string baseMacroName = getBaseMacroName();
  std::string macroName = getMacroName(kind);
  StringRef macroArgs = getMacroArgs(kind);

  os << "#ifndef " << macroName << "\n";
  os << "#define " << macroName << macroArgs << " ";
  os << baseMacroName << "(NAME, " << kind.type << ", TAPIRONLY, IRNAME)\n";
  os << "#endif // " << macroName << "\n";
  os << "\n";
}

void KitLoopAttrEmitter::emitAttr(raw_ostream &os, const Record &attr) {
  const Record *type = attr.getValueAsDef("ValueType");
  Kind kind = getKind(*type);
  std::string macroName = getMacroName(kind);
  std::string irName = getIRName(attr);
  StringRef attrName = attr.getName();

  os << macroName << "(" << attrName << ", \"" << irName << "\"";
  os << ", " << (isTapirOnly(attr) ? "true" : "false");
  if (kind.name == "ENUM")
    os << ", " << type->getValueAsString("Name");
  os << ")\n";
}

KitLoopAttrEmitter::KitLoopAttrEmitter(const RecordKeeper &records)
    : KitAttrHeaderEmitter(records) {}

static TableGen::Emitter::OptClass<KitLoopAttrEmitter>
    X("gen-kit-loop-attr-header",
      "Generate header for Kitsune-specific loop attributes");
