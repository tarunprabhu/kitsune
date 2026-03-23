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

public:
  KitLoopAttrEmitter(const RecordKeeper &records);
  virtual ~KitLoopAttrEmitter() = default;
};

} // namespace

StringRef KitLoopAttrEmitter::getMacroRoot() const { return "LOOP"; }

StringRef KitLoopAttrEmitter::getIRNamePrefix(const Record &attr) const {
  return "tapir.loop.";
}

StringRef KitLoopAttrEmitter::getAttrBase() const { return "LoopAttr"; }

KitLoopAttrEmitter::KitLoopAttrEmitter(const RecordKeeper &records)
    : KitAttrHeaderEmitter(records) {}

static TableGen::Emitter::OptClass<KitLoopAttrEmitter>
    X("gen-kit-loop-attr-header",
      "Generate header for Kitsune-specific loop attributes");
