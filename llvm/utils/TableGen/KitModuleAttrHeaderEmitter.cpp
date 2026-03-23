//===- KitModuleAttrHeaderEmitter.cpp - Generate header for module attrs --===//
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

#define DEBUG_TYPE "kit-module-attr-header"

using namespace llvm;

namespace {

class KitModuleAttrEmitter : public KitAttrHeaderEmitter {
protected:
  StringRef getMacroRoot() const override;
  StringRef getIRNamePrefix(const Record &attr) const override;
  StringRef getAttrBase() const override;

public:
  KitModuleAttrEmitter(const RecordKeeper &records);
  virtual ~KitModuleAttrEmitter() = default;
};

} // namespace

StringRef KitModuleAttrEmitter::getMacroRoot() const { return "MODULE"; }

StringRef KitModuleAttrEmitter::getIRNamePrefix(const Record &attr) const {
  return "kit.module.";
}

StringRef KitModuleAttrEmitter::getAttrBase() const { return "ModuleAttr"; }

KitModuleAttrEmitter::KitModuleAttrEmitter(const RecordKeeper &records)
    : KitAttrHeaderEmitter(records) {}

static TableGen::Emitter::OptClass<KitModuleAttrEmitter>
    X("gen-kit-module-attr-header",
      "Generate header for Kitsune-specific module attributes");
