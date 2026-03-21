//=- KitFuncAttrHeaderEmitter.cpp - Generate header for function attributes -=//
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

#define DEBUG_TYPE "kit-func-attr-header"

using namespace llvm;

namespace {

class KitFuncAttrEmitter : public KitAttrHeaderEmitter {
protected:
  StringRef getMacroRoot() const override;
  StringRef getIRNamePrefix(const Record &attr) const override;
  StringRef getAttrBase() const override;

public:
  KitFuncAttrEmitter(const RecordKeeper &records);
};

} // namespace

StringRef KitFuncAttrEmitter::getMacroRoot() const { return "FUNC"; }

StringRef KitFuncAttrEmitter::getIRNamePrefix(const Record &attr) const {
  return "kit.func.";
}

StringRef KitFuncAttrEmitter::getAttrBase() const { return "FuncAttr"; }

KitFuncAttrEmitter::KitFuncAttrEmitter(const RecordKeeper &records)
    : KitAttrHeaderEmitter(records) {}

static TableGen::Emitter::OptClass<KitFuncAttrEmitter>
    X("gen-kit-func-attr-header",
      "Generate header for Kitsune-specific function attributes");
