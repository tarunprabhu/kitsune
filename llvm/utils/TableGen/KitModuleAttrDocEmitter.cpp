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

public:
  KitModuleAttrDocEmitter(const RecordKeeper &records);
  virtual ~KitModuleAttrDocEmitter() = default;
};

} // namespace

StringRef KitModuleAttrDocEmitter::getEnumName() const {
  return "ModuleAttrKind";
}

StringRef KitModuleAttrDocEmitter::getAttrBase() const { return "ModuleAttr"; }

StringRef KitModuleAttrDocEmitter::getLabelPrefix() const { return "module"; }

StringRef KitModuleAttrDocEmitter::getIRNamePrefix(const Record &attr) const {
  return "kit.module.";
}

KitModuleAttrDocEmitter::KitModuleAttrDocEmitter(const RecordKeeper &records)
    : KitAttrDocEmitter(records) {}

static TableGen::Emitter::OptClass<KitModuleAttrDocEmitter>
    X("gen-kit-module-attr-doc",
      "Generate documentation for Kitsune-specific module attributes");
