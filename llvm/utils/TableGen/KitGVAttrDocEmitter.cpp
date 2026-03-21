//=- KitGVAttrDocEmitter.cpp - Generate docs for global variable attributes -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KitAttrCommon.h"
#include "KitAttrDocEmitter.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kit-gv-attr-doc"

using namespace llvm;

namespace {

class KitGVAttrDocEmitter : public KitAttrDocEmitter {
protected:
  virtual StringRef getEnumName() const override;
  virtual StringRef getAttrBase() const override;
  virtual StringRef getLabelPrefix() const override;
  virtual StringRef getIRNamePrefix(const Record &attr) const override;

public:
  KitGVAttrDocEmitter(const RecordKeeper &records);
};

} // namespace

StringRef KitGVAttrDocEmitter::getEnumName() const { return "GVAttrKind"; }

StringRef KitGVAttrDocEmitter::getAttrBase() const { return "GVAttr"; }

StringRef KitGVAttrDocEmitter::getLabelPrefix() const { return "gv"; }

StringRef KitGVAttrDocEmitter::getIRNamePrefix(const Record &attr) const {
  return "kit.gv.";
}

KitGVAttrDocEmitter::KitGVAttrDocEmitter(const RecordKeeper &records)
    : KitAttrDocEmitter(records) {}

static TableGen::Emitter::OptClass<KitGVAttrDocEmitter>
    X("gen-kit-gv-attr-doc",
      "Generate documentation for Kitsune-specific global variable attributes");
