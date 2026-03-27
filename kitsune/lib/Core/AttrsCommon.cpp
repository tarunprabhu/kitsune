//===- AttrsCommon.cpp - Utilities common to Kitsune-specific attributes --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities shared by Kitsune-specific attributes.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/AttrsCommon.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

// All Kitsune's attributes follow the pattern of LLVM's loop attributes.
// Consider the example below:
//
//     define void @f() {
//         ret void, !kit.inst.attrs !0
//     }
//
// !0 = distinct !{!0, !1, !2}
// !1 = !{!"attr-name"}
// !2 = !{!"some-other-attr-name", ...}
//
// Here, the return instruction has two attributes - one named "attr-name", the
// other named "some-other-attr-name". The former does not have any values.
// The latter has one or more attributes indicated by ellipses (in LLVM-IR,
// these will usually be ConstantAsMetadata, but they may also be MDNode's).
//
// These attributes are elements of a distinct "attribute list" MDNode. The list
// is self-referential. The first operand will always be a reference to itself.

// Create a new MDNode that will act as an attribute list with the given
// attributes.
static MDNode *makeAttrList(LLVMContext &ctx, ArrayRef<Metadata *> attrs) {
  MDNode *md = MDNode::getDistinct(ctx, attrs);
  md->replaceOperandWith(0, md);
  return md;
}

static std::optional<unsigned> getAttrIndex(StringRef attrName,
                                            const MDNode *attrList) {
  if (attrList)
    for (unsigned i = 1, e = attrList->getNumOperands(); i < e; ++i)
      if (auto *md = dyn_cast<MDNode>(attrList->getOperand(i)))
        if (auto *mdStr = dyn_cast<MDString>(md->getOperand(0)))
          if (mdStr->getString() == attrName)
            return i;
  return std::nullopt;
}

// Replace the element at index \p i in the attribute list \p attrList with
// \p attr. Return \p attrList.
static MDNode *replaceInAttrList(MDNode *attrList, unsigned i, Metadata *attr) {
  assert(attrList && "Cannot replace element of null list");
  assert(i < attrList->getNumOperands() && "Invalid index in attribute list");

  attrList->replaceOperandWith(i, attr);
  return attrList;
}

// Append the element \p attr to the attribute list \p attrList. Return the
// newly created attribute list. If \p attrList is nullptr, create a singleton
// list containing only \p attr.
static MDNode *appendToAttrList(MDNode *attrList, MDNode *attr) {
  // The first element of the new attribute list must be a self-reference. It
  // will be replaced when the new attribute list is created.
  SmallVector<Metadata *, 8> newAttrs = {nullptr};
  if (attrList)
    for (Metadata *op : attrList->operands().drop_front())
      newAttrs.push_back(op);
  newAttrs.push_back(attr);

  return makeAttrList(attr->getContext(), newAttrs);
}

MDNode *llvm::makeRawAttr(LLVMContext &ctx, StringRef attrName,
                          ArrayRef<Metadata *> attrVals) {
  SmallVector<Metadata *, 8> ops;
  MDString *mdName = MDString::get(ctx, attrName);

  ops.push_back(mdName);
  ops.append(attrVals.begin(), attrVals.end());

  return MDNode::get(ctx, ops);
}

MDNode *llvm::getNewAttrList(LLVMContext &ctx) {
  return makeAttrList(ctx, {nullptr});
}

MDNode *llvm::getAttrListWith(StringRef attrName,
                              const ArrayRef<Metadata *> attrVals,
                              MDNode *attrList, LLVMContext &ctx) {
  MDNode *attr = makeRawAttr(ctx, attrName, attrVals);
  if (std::optional<unsigned> i = getAttrIndex(attrName, attrList))
    return replaceInAttrList(attrList, *i, attr);
  else
    return appendToAttrList(attrList, attr);
}

MDNode *llvm::getAttrListWithout(StringRef attrName, MDNode *attrList) {
  if (!attrList)
    return nullptr;

  LLVMContext &ctx = attrList->getContext();

  // Since we will always create a new attribute list node, the first element
  // must be a self-reference. It will be replaced when the new attribute list
  // is created.
  SmallVector<Metadata *, 8> newAttrs = {nullptr};
  for (Metadata *op : attrList->operands().drop_front())
    if (auto *md = dyn_cast<MDNode>(op))
      if (auto *mdStr = dyn_cast<MDString>(md->getOperand(0)))
        if (mdStr->getString() != attrName)
          newAttrs.push_back(md);

  if (newAttrs.size() == 1)
    return nullptr;
  else if (newAttrs.size() == attrList->getNumOperands())
    return attrList;
  else
    return makeAttrList(ctx, newAttrs);
}

MDNode *llvm::getRawAttr(StringRef attrName, const MDNode *attrList) {
  if (attrList)
    for (Metadata *op : attrList->operands().drop_front())
      if (auto *md = dyn_cast<MDNode>(op))
        if (md->getNumOperands())
          if (auto *mdStr = dyn_cast<MDString>(md->getOperand(0)))
            if (mdStr->getString() == attrName)
              return md;
  return nullptr;
}

std::optional<Loop *>
llvm::getAttrValue(StringRef attrName, const MDNode *attrList,
                   const SmallVectorImpl<const LoopInfo *> &lis) {
  if (MDNode *md = getRawAttr(attrName, attrList))
    if (md->getNumOperands() == 2)
      for (const LoopInfo *li : lis)
        for (Loop *loop : *li)
          if (md->getOperand(1).get() == loop->getLoopID())
            return loop;
  return std::nullopt;
}

bool llvm::verifyAttr0(StringRef attrName, const MDNode *attrList,
                       raw_ostream *os) {
  if (MDNode *md = getRawAttr(attrName, attrList)) {
    if (md->getNumOperands() != 1) {
      if (os)
        (*os) << "Unexpected value in attribute '" << attrName << "'\n";
      return false;
    }
  }
  return true;
}

bool llvm::verifyAttrLoop(StringRef attrName, const MDNode *attrList,
                          raw_ostream *os) {
  auto printError = [](StringRef attrName, raw_ostream *os) -> bool {
    if (os)
      (*os) << "Missing value of type 'Loop' in attribute '" << attrName
            << "'\n";
    return false;
  };

  if (MDNode *md = getRawAttr(attrName, attrList)) {
    if (md->getNumOperands() != 2)
      return printError(attrName, os);

    MDNode *val = dyn_cast<MDNode>(md->getOperand(1));
    if (!val || !val->getNumOperands() || !val->isDistinct() ||
        val->getOperand(0) != val)
      return printError(attrName, os);
  }
  return true;
}
