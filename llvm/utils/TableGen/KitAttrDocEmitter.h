//===- KitAttrDocEmitter.h - Base class to emit attribute docs ---*-C++-*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for emitters that generate documentation for Kitsune-specific
// attributes
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_KIT_ATTR_DOC_EMITTER_H
#define LLVM_TABLEGEN_KIT_ATTR_DOC_EMITTER_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class Record;
class RecordKeeper;
class raw_ostream;

} // namespace llvm

class KitAttrDocEmitter {
protected:
  const llvm::RecordKeeper &records;

protected:
  KitAttrDocEmitter(const llvm::RecordKeeper &records);

  std::string quote(llvm::StringRef s, llvm::StringRef q = "\"") const;
  std::string getSectionLabel(const llvm::Record &attr) const;
  std::string getEnum(const llvm::Record &attr) const;
  std::string getValueType(const llvm::Record &attr) const;

  virtual llvm::StringRef getEnumName() const = 0;
  virtual llvm::StringRef getAttrBase() const = 0;
  virtual llvm::StringRef getLabelPrefix() const = 0;
  virtual llvm::StringRef getIRNamePrefix(const llvm::Record &attr) const = 0;

  virtual void emitAttrHeader(llvm::raw_ostream &os, const llvm::Record &attr);
  virtual void emitAttrArgs(llvm::raw_ostream &os, const llvm::Record &attr);
  virtual void emitAttrDoc(llvm::raw_ostream &os, const llvm::Record &attr);

public:
  virtual ~KitAttrDocEmitter() = default;

  void run(llvm::raw_ostream &os);
};

#endif // LLVM_TABLEGEN_KIT_ATTR_DOC_EMITTER_H
