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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Instruction;
class MDNode;
class Metadata;

namespace detail {

MDNode *getRawAttrList(const Instruction &inst);
void setAttrList(Instruction &inst, MDNode *attrList);
void addAttr(Instruction &inst, StringRef attrName,
             ArrayRef<Metadata *> attrVals);
void removeAttr(Instruction &inst, StringRef attrName);

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_CORE_INST_ATTRS_IMPL_H
