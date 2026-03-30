//===- LoopAttrsImpl.h -  Kitsune-specific loop attributes ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private header with declarations for the core implementation of
// Kitsune-specific loop attributes. These intentionally "leak" so they can be
// shared with the unit tests.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_CORE_LOOP_ATTRS_IMPL_H
#define KITSUNE_LIB_CORE_LOOP_ATTRS_IMPL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Loop;
class MDNode;
class Metadata;

namespace detail {

MDNode *getRawAttrList(const Loop &loop);
void setAttrList(Loop &loop, MDNode *attrList);
void addAttr(Loop &loop, StringRef attrName, ArrayRef<Metadata *> attrVals);
void removeAttr(Loop &loop, StringRef attrName);

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_CORE_LOOP_ATTRS_IMPL_H
