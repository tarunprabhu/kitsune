//===- FuncAttrsImpl.h -  Kitsune-specific function attributes --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private header with declarations for the core implementation of
// Kitsune-specific function attributes. These intentionally "leak" so they can
// be shared with the unit tests.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_CORE_FUNC_ATTRS_IMPL_H
#define KITSUNE_LIB_CORE_FUNC_ATTRS_IMPL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Function;
class MDNode;
class Metadata;

namespace detail {

MDNode *getRawAttrList(const Function &f);
void setAttrList(Function &f, MDNode *attrList);
void addAttr(Function &f, StringRef attrName, ArrayRef<Metadata *> attrVals);
void removeAttr(Function &f, StringRef attrName);

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_CORE_FUNC_ATTRS_IMPL_H
