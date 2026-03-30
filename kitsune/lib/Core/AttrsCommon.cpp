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
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/Verifier.h"
#include "kitsune/Support/Diagnostics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

bool llvm::detail::verifyRawAttrValueCount(KitVerifier &v, const MDNode &attr,
                                           unsigned attrVals) {
  StringRef attrName = getRawAttrName(attr);
  unsigned numVals = attr.getNumOperands() - 1;
  return v.check(numVals == attrVals, DiagID::ErrAttrBadValues, attrName,
                 numVals, attrVals);
}

bool llvm::detail::verifyRawAttrValueLoop(KitVerifier &v, const MDNode &attr) {
  StringRef attrName = getRawAttrName(attr);
  unsigned n = attr.getNumOperands();
  if (!v.check(n == 2, DiagID::ErrAttrBadValues, attrName, n - 1, 1))
    return false;

  MDNode *val = dyn_cast<MDNode>(attr.getOperand(1));
  bool isLoop = val && val->getNumOperands() && val->isDistinct();
  if (!v.check(isLoop, DiagID::ErrAttrBadValue, attrName,
               "MDNode is not a valid loop id"))
    return false;

  return true;
}

template <typename T>
bool llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &attr,
                                        unsigned i,
                                        const std::optional<T> &val) {
  StringRef attrName = cast<MDString>(attr.getOperand(0))->getString();
  return v.check(val.has_value(), DiagID::ErrAttrNoValueAt, attrName,
                 toString<T>(), i);
}

template bool
llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &, unsigned i,
                                   const std::optional<int8_t> &val);
template bool
llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &, unsigned i,
                                   const std::optional<uint8_t> &val);
template bool
llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &, unsigned i,
                                   const std::optional<int16_t> &val);
template bool
llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &, unsigned i,
                                   const std::optional<uint16_t> &val);
template bool
llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &, unsigned i,
                                   const std::optional<int32_t> &val);
template bool
llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &, unsigned i,
                                   const std::optional<uint32_t> &val);
template bool
llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &, unsigned i,
                                   const std::optional<int64_t> &val);
template bool
llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &, unsigned i,
                                   const std::optional<uint64_t> &val);
template bool
llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &, unsigned i,
                                   const std::optional<float> &val);
template bool
llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &, unsigned i,
                                   const std::optional<double> &val);
template bool
llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &, unsigned i,
                                   const std::optional<StringRef> &val);
template bool
llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &, unsigned i,
                                   const std::optional<TTID> &val);
template bool
llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &, unsigned i,
                                   const std::optional<TapirSpawnStrategy> &);

// -----------------------------------------------------------------------------

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

StringRef llvm::getRawAttrName(const MDNode &attr) {
  return cast<MDString>(attr.getOperand(0))->getString();
}

std::optional<Loop *>
llvm::getRawAttrValue(const MDNode &attr,
                      const SmallVectorImpl<const LoopInfo *> &lis) {
  if (attr.getNumOperands() == 2) {
    Metadata *val = attr.getOperand(1);
    for (const LoopInfo *li : lis)
      for (Loop *loop : *li)
        if (val == loop->getLoopID())
          return loop;
  }
  return std::nullopt;
}

template <typename T>
std::optional<T> llvm::getRawAttrValue(const MDNode &attr, size_t i) {
  // The first operand of the attribute will be the name of the attribute.
  unsigned attrIdx = i + 1;
  if (attrIdx < attr.getNumOperands())
    return fromMetadata<T>(attr.getOperand(attrIdx));
  return std::nullopt;
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
  if (MDNode *attr = getRawAttr(attrName, attrList))
    return getRawAttrValue(*attr, lis);
  return std::nullopt;
}

template <typename T>
std::optional<T> llvm::getAttrValue(StringRef attrName, const MDNode *attrList,
                                    unsigned valNo, unsigned vals) {
  if (attrList && vals && valNo < vals)
    if (MDNode *attr = getRawAttr(attrName, attrList))
      // The first operand of the metadata node will be the name of the
      // attribute.
      if (attr->getNumOperands() == vals + 1)
        return fromMetadata<T>(attr->getOperand(valNo + 1));
  return std::nullopt;
}

// We only support a limited number of types that can be in metadata, so just
// instantiate everything explicitly. All enums have to be explicitly
// instantiated as well. This is not unreasonable because we don't expect to
// support completely arbitrary enums as attribute values.

template std::optional<int8_t> llvm::getRawAttrValue(const MDNode &, size_t);
template std::optional<uint8_t> llvm::getRawAttrValue(const MDNode &, size_t);
template std::optional<int16_t> llvm::getRawAttrValue(const MDNode &, size_t);
template std::optional<uint16_t> llvm::getRawAttrValue(const MDNode &, size_t);
template std::optional<int32_t> llvm::getRawAttrValue(const MDNode &, size_t);
template std::optional<uint32_t> llvm::getRawAttrValue(const MDNode &, size_t);
template std::optional<int64_t> llvm::getRawAttrValue(const MDNode &, size_t);
template std::optional<uint64_t> llvm::getRawAttrValue(const MDNode &, size_t);
template std::optional<float> llvm::getRawAttrValue(const MDNode &, size_t);
template std::optional<double> llvm::getRawAttrValue(const MDNode &, size_t);
template std::optional<StringRef> llvm::getRawAttrValue(const MDNode &, size_t);
template std::optional<TTID> llvm::getRawAttrValue(const MDNode &, size_t);
template std::optional<TapirSpawnStrategy> llvm::getRawAttrValue(const MDNode &,
                                                                 size_t);

template std::optional<int8_t> llvm::getAttrValue(StringRef, const MDNode *,
                                                  unsigned, unsigned);
template std::optional<uint8_t> llvm::getAttrValue(StringRef, const MDNode *,
                                                   unsigned, unsigned);
template std::optional<int16_t> llvm::getAttrValue(StringRef, const MDNode *,
                                                   unsigned, unsigned);
template std::optional<uint16_t> llvm::getAttrValue(StringRef, const MDNode *,
                                                    unsigned, unsigned);
template std::optional<int32_t> llvm::getAttrValue(StringRef, const MDNode *,
                                                   unsigned, unsigned);
template std::optional<uint32_t> llvm::getAttrValue(StringRef, const MDNode *,
                                                    unsigned, unsigned);
template std::optional<int64_t> llvm::getAttrValue(StringRef, const MDNode *,
                                                   unsigned, unsigned);
template std::optional<uint64_t> llvm::getAttrValue(StringRef, const MDNode *,
                                                    unsigned, unsigned);
template std::optional<float> llvm::getAttrValue(StringRef, const MDNode *,
                                                 unsigned, unsigned);
template std::optional<double> llvm::getAttrValue(StringRef, const MDNode *,
                                                  unsigned, unsigned);
template std::optional<StringRef> llvm::getAttrValue(StringRef, const MDNode *,
                                                     unsigned, unsigned);
template std::optional<TTID> llvm::getAttrValue(StringRef, const MDNode *,
                                                unsigned, unsigned);
template std::optional<TapirSpawnStrategy>
llvm::getAttrValue(StringRef, const MDNode *, unsigned, unsigned);
