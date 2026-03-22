//==- KitInstAttrHeaderEmitter.cpp - Generate header for instruction attrs -==//
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

#define DEBUG_TYPE "kit-inst-attr-header"

using namespace llvm;

namespace {

class KitInstAttrEmitter : public KitAttrHeaderEmitter {
protected:
  StringRef getMacroRoot() const override;
  StringRef getIRNamePrefix(const Record &attr) const override;
  StringRef getAttrBase() const override;

public:
  KitInstAttrEmitter(const RecordKeeper &records);
  virtual ~KitInstAttrEmitter() = default;
};

} // namespace

StringRef KitInstAttrEmitter::getMacroRoot() const { return "INST"; }

StringRef KitInstAttrEmitter::getIRNamePrefix(const Record &attr) const {
  return "kit.inst.";
}

StringRef KitInstAttrEmitter::getAttrBase() const { return "InstAttr"; }

KitInstAttrEmitter::KitInstAttrEmitter(const RecordKeeper &records)
    : KitAttrHeaderEmitter(records) {}

static TableGen::Emitter::OptClass<KitInstAttrEmitter>
    X("gen-kit-inst-attr-header",
      "Generate header for Kitsune-specific instruction attributes");
