//===- KitArgAttrDocEmitter.cpp - Generate docs for argument attributes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KitAttrCommon.h"
#include "KitAttrDocEmitter.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kit-arg-attr-doc"

using namespace llvm;

namespace {

class KitArgAttrDocEmitter : public KitAttrDocEmitter {
protected:
  virtual StringRef getEnumName() const override;
  virtual StringRef getAttrBase() const override;
  virtual StringRef getLabelPrefix() const override;
  virtual StringRef getIRNamePrefix(const Record &attr) const override;

public:
  KitArgAttrDocEmitter(const RecordKeeper &records);
  virtual ~KitArgAttrDocEmitter() = default;
};

} // namespace

StringRef KitArgAttrDocEmitter::getEnumName() const { return "ArgAttrKind"; }

StringRef KitArgAttrDocEmitter::getAttrBase() const { return "ArgAttr"; }

StringRef KitArgAttrDocEmitter::getLabelPrefix() const { return "arg"; }

StringRef KitArgAttrDocEmitter::getIRNamePrefix(const Record &attr) const {
  return "kit.arg.";
}

KitArgAttrDocEmitter::KitArgAttrDocEmitter(const RecordKeeper &records)
    : KitAttrDocEmitter(records) {}

static TableGen::Emitter::OptClass<KitArgAttrDocEmitter>
    X("gen-kit-arg-attr-doc",
      "Generate documentation for Kitsune-specific argument attributes");
