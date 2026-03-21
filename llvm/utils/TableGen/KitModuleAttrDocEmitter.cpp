//===- KitModuleAttrDocEmitter.cpp - Generate docs for module attributes --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KitAttrCommon.h"
#include "KitAttrDocEmitter.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kit-module-attr-doc"

using namespace llvm;

namespace {

class KitModuleAttrDocEmitter : public KitAttrDocEmitter {
protected:
  virtual StringRef getEnumName() const override;
  virtual StringRef getAttrBase() const override;
  virtual StringRef getLabelPrefix() const override;
  virtual StringRef getIRNamePrefix(const Record &attr) const override;

  virtual void emitAttrArgs(raw_ostream &os, const Record &attr) override;

public:
  KitModuleAttrDocEmitter(const RecordKeeper &records);
};

} // namespace

StringRef KitModuleAttrDocEmitter::getEnumName() const {
  return "ModuleAttrKind";
}

StringRef KitModuleAttrDocEmitter::getAttrBase() const { return "ModuleAttr"; }

StringRef KitModuleAttrDocEmitter::getLabelPrefix() const { return "module"; }

StringRef KitModuleAttrDocEmitter::getIRNamePrefix(const Record &attr) const {
  return "kit.module";
}

void KitModuleAttrDocEmitter::emitAttrArgs(raw_ostream &os,
                                           const Record &attr) {
  os << ".. csv-table::\n";
  os << "  :header: " << quote("Enum") << ", " << quote("Value Types") << "\n";
  os << "\n";
  os << "  " << getEnum(attr) << ", " << quote("TODO") << "\n";
  os << "\n";
}

KitModuleAttrDocEmitter::KitModuleAttrDocEmitter(const RecordKeeper &records)
    : KitAttrDocEmitter(records) {}

static TableGen::Emitter::OptClass<KitModuleAttrDocEmitter>
    X("gen-kit-module-attr-doc",
      "Generate documentation for Kitsune-specific module attributes");
