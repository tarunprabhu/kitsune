//=- KitGVAttrHeaderEmitter.cpp - Generate header for global var attributes -=//
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

#define DEBUG_TYPE "kit-gv-attr-header"

using namespace llvm;

namespace {

class KitGVAttrEmitter : public KitAttrHeaderEmitter {
protected:
  StringRef getMacroRoot() const override;
  StringRef getIRNamePrefix(const Record &attr) const override;
  StringRef getAttrBase() const override;

public:
  KitGVAttrEmitter(const RecordKeeper &records);
};

} // namespace

StringRef KitGVAttrEmitter::getMacroRoot() const { return "GV"; }

StringRef KitGVAttrEmitter::getIRNamePrefix(const Record &attr) const {
  return "kit.gv.";
}

StringRef KitGVAttrEmitter::getAttrBase() const { return "GVAttr"; }

KitGVAttrEmitter::KitGVAttrEmitter(const RecordKeeper &records)
    : KitAttrHeaderEmitter(records) {}

static TableGen::Emitter::OptClass<KitGVAttrEmitter>
    X("gen-kit-gv-attr-header",
      "Generate header for Kitsune-specific global variable attributes");
