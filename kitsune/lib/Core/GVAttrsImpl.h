//=- GVAttrsImpl.h -  Kitsune-specific global variable attributes -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private header with declarations for the core implementation of
// Kitsune-specific global variable attributes. These intentionally "leak" so
// they can be shared with the unit tests.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_CORE_GV_ATTRS_IMPL_H
#define KITSUNE_LIB_CORE_GV_ATTRS_IMPL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class GlobalVariable;
class MDNode;
class Metadata;

namespace detail {

MDNode *getRawAttrList(const GlobalVariable &g);
void setAttrList(GlobalVariable &g, MDNode *attrList);
void addAttr(GlobalVariable &g, StringRef attrName,
             ArrayRef<Metadata *> attrVals);
void removeAttr(GlobalVariable &g, StringRef attrName);

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_CORE_GV_ATTRS_IMPL_H
