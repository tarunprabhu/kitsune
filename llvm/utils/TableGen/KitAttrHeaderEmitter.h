//=- KitAttrHeaderEmitter.h - Base class to emit attribute headers -*-C++-*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for emitters that generate headers for Kitsune-specific attributes
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_KIT_ATTR_HEADER_EMITTER_H
#define LLVM_TABLEGEN_KIT_ATTR_HEADER_EMITTER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Record;
class RecordKeeper;
class raw_ostream;

} // namespace llvm

/// Base class for emitters that generate headers for Kitsune's attributes. In
/// general, the header files for attributes for a given IR unit have the
/// following structure.
///
///     #ifndef <GUARD_ATTRS>
///     #define <GUARD_ATTRS>
///
///     #ifndef <BASE_MACRO_NAME>
///     #define <BASE_MACRO_NAME>(<BASE_MACRO_ARGS>)
///     #endif // <BASE_MACRO_NAME>
///
///     #ifndef <MACRO_NAME_1>
///     #define <MACRO_NAME_1>(<MACRO_ARGS>)  <BASE_MACRO_NAME>(...)
///     #endif // <MACRO_NAME_1>
///
///     ...
///
///     #ifndef <MACRO_NAME_N>
///     #define <MACRO_NAME_N>(<MACRO_ARGS>)  <BASE_MACRO_NAME>(...)
///     #endif // <MACRO_NAME_N>
///
///     // Declarations of the attributes. Each will use one of the
///     // <MACRO_NAME_*> macros.
///     <MACRO_NAME_*>(...)
///     ...
///
///     #undef <MACRO_NAME_N>
///     ...
///     #undef <MACRO_NAME_1>
///     #endif // <GUARD_ATTRS>
///
///
///     #ifndef <GUARD_ENUMS>
///     #define <GUARD_ENUMS>
///
///     EnumElement1 = 1,
///     ...
///     EnumElementN = N
///
///     #endif // <GUARD_ENUMS>
///
/// The table below summarizes the structure of the names of various macros and
/// guards.
///
///     <GUARD_ATTRS>        GET_<MACRO_ROOT>_ATTRS
///     <BASE_MACRO_NAME>    <MACRO_ROOT>_ATTR
///     <MACRO_NAME_1>       <MACRO_ROOT>_ATTR_<KIND_1>
///     ...
///     <MACRO_NAME_N>       <MACRO_ROOT>_ATTR_<KIND_N>
///     <GUARD_ENUMS>        GET_<MACRO_ROOT>_ENUMS
///
class KitAttrHeaderEmitter {
protected:
  /// The kind of an attribute. The kind depends on the argument, if any, that
  /// the argument takes. The "name" of the kind will be entirely in uppercase
  /// and may be a type "class". For instance, all attributes that take an enum
  /// value as an argument will have the name "ENUM". The type is the actual C++
  /// type of the argument. For instance, attributes ATTR1 and ATTR2 that
  /// take arguments `enum TTID` and `enum SpawnStrategy` will both have name
  /// "ENUM", but type's "TTID" and "SpawnStrategy" respectively.
  struct Kind {
    std::string name;
    llvm::StringRef type;

    friend bool operator==(const Kind &l, const Kind &r) {
      return l.name == r.name;
    }

    friend bool operator<(const Kind &l, const Kind &r) {
      return l.name < r.name;
    }
  };

protected:
  /// All records in the .td file being processed. This will include records
  /// from any included files as well.
  const llvm::RecordKeeper &records;

  /// The "kinds" of attributes. An attribute's kind depends on the argument,
  /// if any, that the attribute takes.
  llvm::SmallVector<Kind, 8> attrKinds;

protected:
  KitAttrHeaderEmitter(const llvm::RecordKeeper &records);

  /// Get the kind of an attribute.
  Kind getKind(const llvm::Record &attr) const;

  /// Get the <MACRO_ROOT>
  virtual llvm::StringRef getMacroRoot() const = 0;

  /// Get the prefix of the attributes for a specific IR element as the will
  /// appear in LLVM-IR. The table below summarizes what these might be. The
  /// table below is just an example. The actual prefixes in the IR may be
  /// different.
  ///
  ///     IR Unit         |  Prefix
  ///     ------------------------------
  ///     Function        |  kit.func.
  ///     GlobalVariable  |  kit.gv.
  ///     Instruction     |  kit.inst.
  ///     Loop            |  tapir.loop.
  ///     Module          |  kit.module.
  ///
  virtual llvm::StringRef getIRNamePrefix(const llvm::Record &attr) const = 0;

  /// The common base class from which all attributes for a given IR unit
  /// inherit. This will usually be defined in the .td file being processed.
  virtual llvm::StringRef getAttrBase() const;

  /// Get the <BASE_MACRO_NAME>
  virtual std::string getBaseMacroName() const;

  /// Get the <BASE_MACRO_ARGS>
  virtual llvm::StringRef getBaseMacroArgs() const;

  /// Get the <MACRO_NAME_*> for the given KIND.
  virtual std::string getMacroName(const Kind &kind) const;

  /// Get the <MACRO_NAME_*> for an attribute.
  virtual std::string getMacroName(const llvm::Record &attr) const;

  /// Get the arguments for <MACRO_NAME_*> for the given KIND.
  virtual llvm::StringRef getMacroArgs(const Kind &kind) const;

  /// Get the full name of the attribute as it will appear in LLVM-IR.
  virtual std::string getIRName(const llvm::Record &attr) const;

  /// Emit the definition of a macro for the given kind.
  virtual void emitMacroDefn(llvm::raw_ostream &os, const Kind &kind);

  virtual void emitAttrsGuardIn(llvm::raw_ostream &os);
  virtual void emitBaseMacroDef(llvm::raw_ostream &os);
  virtual void emitMacroDefs(llvm::raw_ostream &os);
  virtual void emitAttr(llvm::raw_ostream &os, const llvm::Record &attr);
  virtual void emitAttrs(llvm::raw_ostream &os);
  virtual void emitMacroUndefs(llvm::raw_ostream &os);
  virtual void emitBaseMacroUndef(llvm::raw_ostream &os);
  virtual void emitAttrsGuardOut(llvm::raw_ostream &os);

  virtual void emitEnums(llvm::raw_ostream &os);

public:
  virtual ~KitAttrHeaderEmitter() = default;

  void run(llvm::raw_ostream &os);
};

#endif // LLVM_TABLEGEN_KIT_ATTR_HEADER_EMITTER_H
