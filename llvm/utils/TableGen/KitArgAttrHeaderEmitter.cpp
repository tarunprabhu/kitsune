//=- KitArgAttrHeaderEmitter.cpp - Generate header for argument attributes --=//
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

#define DEBUG_TYPE "kit-arg-attr-header"

using namespace llvm;

namespace {

class KitArgAttrEmitter : public KitAttrHeaderEmitter {
protected:
  StringRef getMacroRoot() const override;
  StringRef getIRNamePrefix(const Record &attr) const override;
  StringRef getAttrBase() const override;

public:
  KitArgAttrEmitter(const RecordKeeper &records);
  virtual ~KitArgAttrEmitter() = default;
};

} // namespace

StringRef KitArgAttrEmitter::getMacroRoot() const { return "ARG"; }

StringRef KitArgAttrEmitter::getIRNamePrefix(const Record &attr) const {
  return "kit.arg.";
}

StringRef KitArgAttrEmitter::getAttrBase() const { return "ArgAttr"; }

KitArgAttrEmitter::KitArgAttrEmitter(const RecordKeeper &records)
    : KitAttrHeaderEmitter(records) {}

static TableGen::Emitter::OptClass<KitArgAttrEmitter>
    X("gen-kit-arg-attr-header",
      "Generate header for Kitsune-specific argument attributes");
