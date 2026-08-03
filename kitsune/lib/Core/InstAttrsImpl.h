//==- InstAttrsImpl.h -  Kitsune-specific instruction attributes -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private header with declarations for the core implementation of
// Kitsune-specific instruction attributes. These intentionally "leak" so they
// can be shared with the unit tests.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_CORE_INST_ATTRS_IMPL_H
#define KITSUNE_LIB_CORE_INST_ATTRS_IMPL_H

#include "AttrsImplDeclMacros.h"
#include "AttrsIterator.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"

namespace llvm {

class Instruction;
class MDNode;
class Metadata;

class KitVerifier;

namespace detail {

/// Get the MDNode, if it exists, for an attribute named \p attrName for the
/// Instruction \p inst. Otherwise, return nullptr.
MDNode *getRawAttr(const Instruction &inst, StringRef attrName);

/// Get the raw list of Kitsune-specific attributes from an instruction.
MDNode *getRawAttrList(const Instruction &inst);

/// Update the raw list of Kitsune-specific attributes in an instruction with
/// the new list \p attrList. If \p attrList is nullptr, the attribute list will
/// be removed from the instruction.
void setAttrList(Instruction &inst, MDNode *attrList);

/// Add an attribute named \p attrName with values \p attrVals to an
/// instruction.
void addAttr(Instruction &inst, StringRef attrName,
             ArrayRef<Metadata *> attrVals);

/// Remove the attribute named \p attrName from an instruction. If the attribute
/// is not already on the instruction, this has no effect.
void removeAttr(Instruction &inst, StringRef attrName);

/// Get a pair of iterators over the raw Kitsune-specific instruction
/// attributes.
iterator_range<AttrIterator> attrs(const Instruction &inst);

/// Verify the attribute named \p attrName that is known to be on the
/// instruction \p inst. This will only set the internal state in the verifier
/// \p v.
void verifyAttr(KitVerifier &v, const Instruction &inst, StringRef attrName);

} // namespace detail

#define INST_ATTR(...) DECL_ATTR_IMPL_COMMON(Instruction, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_L(...) DECL_ATTR_IMPL_L(Instruction, __VA_ARGS__)
#define INST_ATTR_S(...) DECL_ATTR_IMPL_S(Instruction, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_0(...) DECL_ATTR_IMPL_0(Instruction, __VA_ARGS__)
#define INST_ATTR_1(...) DECL_ATTR_IMPL_1(Instruction, __VA_ARGS__)
#define INST_ATTR_2(...) DECL_ATTR_IMPL_2(Instruction, __VA_ARGS__)
#define INST_ATTR_3(...) DECL_ATTR_IMPL_3(Instruction, __VA_ARGS__)
#define INST_ATTR_4(...) DECL_ATTR_IMPL_4(Instruction, __VA_ARGS__)
#define INST_ATTR_5(...) DECL_ATTR_IMPL_5(Instruction, __VA_ARGS__)
#define INST_ATTR_6(...) DECL_ATTR_IMPL_6(Instruction, __VA_ARGS__)
#define INST_ATTR_7(...) DECL_ATTR_IMPL_7(Instruction, __VA_ARGS__)
#define INST_ATTR_8(...) DECL_ATTR_IMPL_8(Instruction, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

} // namespace llvm

#endif // KITSUNE_LIB_CORE_INST_ATTRS_IMPL_H
