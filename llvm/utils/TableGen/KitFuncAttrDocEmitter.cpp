//===- KitFuncAttrDocEmitter.cpp - Generate docs for function attributes --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KitAttrCommon.h"
#include "KitAttrDocEmitter.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kit-func-attr-doc"

using namespace llvm;

namespace {

class KitFuncAttrDocEmitter : public KitAttrDocEmitter {
protected:
  virtual StringRef getEnumName() const override;
  virtual StringRef getAttrBase() const override;
  virtual StringRef getLabelPrefix() const override;
  virtual StringRef getIRNamePrefix(const Record &attr) const override;

public:
  KitFuncAttrDocEmitter(const RecordKeeper &records);
  virtual ~KitFuncAttrDocEmitter() = default;
};

} // namespace

StringRef KitFuncAttrDocEmitter::getEnumName() const { return "FuncAttrKind"; }

StringRef KitFuncAttrDocEmitter::getAttrBase() const { return "FuncAttr"; }

StringRef KitFuncAttrDocEmitter::getLabelPrefix() const { return "func"; }

StringRef KitFuncAttrDocEmitter::getIRNamePrefix(const Record &attr) const {
  return "kit.func.";
}

KitFuncAttrDocEmitter::KitFuncAttrDocEmitter(const RecordKeeper &records)
    : KitAttrDocEmitter(records) {}

static TableGen::Emitter::OptClass<KitFuncAttrDocEmitter>
    X("gen-kit-func-attr-doc",
      "Generate documentation for Kitsune-specific function attributes");
