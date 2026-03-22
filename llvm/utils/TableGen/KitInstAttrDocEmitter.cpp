//===- KitInstAttrDocEmitter.cpp - Generate docs for instruction attrs ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KitAttrCommon.h"
#include "KitAttrDocEmitter.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kit-inst-attr-doc"

using namespace llvm;

namespace {

class KitInstAttrDocEmitter : public KitAttrDocEmitter {
protected:
  virtual StringRef getEnumName() const override;
  virtual StringRef getAttrBase() const override;
  virtual StringRef getLabelPrefix() const override;
  virtual StringRef getIRNamePrefix(const Record &attr) const override;

public:
  KitInstAttrDocEmitter(const RecordKeeper &records);
  virtual ~KitInstAttrDocEmitter() = default;
};

} // namespace

StringRef KitInstAttrDocEmitter::getEnumName() const { return "InstAttrKind"; }

StringRef KitInstAttrDocEmitter::getAttrBase() const { return "InstAttr"; }

StringRef KitInstAttrDocEmitter::getLabelPrefix() const { return "inst"; }

StringRef KitInstAttrDocEmitter::getIRNamePrefix(const Record &attr) const {
  return "kit.inst.";
}

KitInstAttrDocEmitter::KitInstAttrDocEmitter(const RecordKeeper &records)
    : KitAttrDocEmitter(records) {}

static TableGen::Emitter::OptClass<KitInstAttrDocEmitter>
    X("gen-kit-inst-attr-doc",
      "Generate documentation for Kitsune-specific instruction attributes");
