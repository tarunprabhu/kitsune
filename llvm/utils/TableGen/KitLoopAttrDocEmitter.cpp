//===- KitLoopAttrDocEmitter.cpp - Generate docs for loop attributes ------===//
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

#define DEBUG_TYPE "kit-loop-attr-doc"

using namespace llvm;

namespace {

class KitLoopAttrDocEmitter : public KitAttrDocEmitter {
protected:
  virtual StringRef getEnumName() const override;
  virtual StringRef getAttrBase() const override;
  virtual StringRef getLabelPrefix() const override;
  virtual StringRef getIRNamePrefix(const Record &attr) const override;

public:
  KitLoopAttrDocEmitter(const RecordKeeper &records);
};

} // namespace

StringRef KitLoopAttrDocEmitter::getEnumName() const { return "LoopAttrKind"; }

StringRef KitLoopAttrDocEmitter::getAttrBase() const { return "LoopAttr"; }

StringRef KitLoopAttrDocEmitter::getLabelPrefix() const { return "loop"; }

StringRef KitLoopAttrDocEmitter::getIRNamePrefix(const Record &attr) const {
  if (isTapirOnly(attr))
    return "tapir.loop.";
  return "loop.";
}

KitLoopAttrDocEmitter::KitLoopAttrDocEmitter(const RecordKeeper &records)
    : KitAttrDocEmitter(records) {}

static TableGen::Emitter::OptClass<KitLoopAttrDocEmitter>
    X("gen-kit-loop-attr-doc",
      "Generate documentation for Kitsune-specific loop attributes");
