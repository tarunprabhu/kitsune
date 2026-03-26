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

static MDNode *makeMDNodeForAttrList(LLVMContext &ctx,
                                     ArrayRef<Metadata *> ops) {
  MDNode *md = MDNode::getDistinct(ctx, ops);
  md->replaceOperandWith(0, md);
  return md;
}

static MDNode *makeMDNodeForAttr(LLVMContext &ctx, StringRef name,
                                 ArrayRef<Metadata *> vals) {
  SmallVector<Metadata *, 8> ops;
  MDString *mdName = MDString::get(ctx, name);

  ops.push_back(mdName);
  ops.append(vals.begin(), vals.end());

  return MDNode::get(ctx, ops);
}

static void copyAttrsExcept(StringRef attrName, const MDNode &attrList,
                            SmallVectorImpl<Metadata *> &newAttrs) {
  for (Metadata *op : attrList.operands().drop_front())
    if (auto *md = dyn_cast<MDNode>(op))
      if (auto *mdStr = dyn_cast<MDString>(md->getOperand(0)))
        if (mdStr->getString() != attrName)
          newAttrs.push_back(md);
}

MDNode *llvm::getNewAttrList(LLVMContext &ctx) {
  return makeMDNodeForAttrList(ctx, {nullptr});
}

MDNode *llvm::getNewAttrListWith(StringRef attrName,
                                 const ArrayRef<Metadata *> attrVals,
                                 const MDNode *attrList, LLVMContext &ctx) {
  // Since we will always create a new attribute list node, the first element
  // must be a self-reference. It will be replaced when the new attribute list
  // is created.
  SmallVector<Metadata *, 8> newAttrs = {nullptr};

  if (attrList)
    copyAttrsExcept(attrName, *attrList, newAttrs);
  newAttrs.push_back(makeMDNodeForAttr(ctx, attrName, attrVals));

  return makeMDNodeForAttrList(ctx, newAttrs);
}

MDNode *llvm::getNewAttrListWithout(StringRef attrName, MDNode *attrList) {
  if (!attrList)
    return nullptr;

  LLVMContext &ctx = attrList->getContext();

  // Since we will always create a new attribute list node, the first element
  // must be a self-reference. It will be replaced when the new attribute list
  // is created.
  SmallVector<Metadata *, 8> newAttrs = {nullptr};
  copyAttrsExcept(attrName, *attrList, newAttrs);

  if (newAttrs.size() == 1)
    return nullptr;
  else if (newAttrs.size() == attrList->getNumOperands())
    return attrList;
  else
    return makeMDNodeForAttrList(ctx, newAttrs);
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
