//===- AttrsImpl.cpp - Core implementation of Kitsune-specific attributes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Core implementation of Kitsune-specific attributes.
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

#include "AttrsImpl.h"
#include "VerifierImpl.h"
#include "kitsune/Core/Diagnostics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

using MaybeI8 = std::optional<int8_t>;
using MaybeU8 = std::optional<uint8_t>;
using MaybeI16 = std::optional<int16_t>;
using MaybeU16 = std::optional<uint16_t>;
using MaybeI32 = std::optional<int32_t>;
using MaybeU32 = std::optional<uint32_t>;
using MaybeI64 = std::optional<int64_t>;
using MaybeU64 = std::optional<uint64_t>;
using MaybeF32 = std::optional<float>;
using MaybeF64 = std::optional<double>;
using MaybeStr = std::optional<StringRef>;
using MaybeMDNode = std::optional<MDNode *>;

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

template <>
bool llvm::detail::verifyRawAttrValueAt(KitVerifier &v, const MDNode &attr,
                                        unsigned i, const MaybeMDNode &val) {
  StringRef attrName = cast<MDString>(attr.getOperand(0))->getString();
  return v.check(val.has_value(), DiagID::ErrAttrNoValueAt, attrName,
                 "llvm::MDNode*", i);
}

MDNode *llvm::detail::makeRawAttr(LLVMContext &ctx, StringRef attrName,
                                  ArrayRef<Metadata *> attrVals) {
  SmallVector<Metadata *, 8> ops;
  MDString *mdName = MDString::get(ctx, attrName);

  ops.push_back(mdName);
  ops.append(attrVals.begin(), attrVals.end());

  return MDNode::get(ctx, ops);
}

template <typename T>
Metadata *llvm::detail::makeRawAttrValue(LLVMContext &ctx, T const &v) {
  return toMetadata<T>(v, ctx);
}

template <>
Metadata *llvm::detail::makeRawAttrValue(LLVMContext &ctx, MDNode *const &v) {
  return v;
}

StringRef llvm::detail::getRawAttrName(const MDNode &attr) {
  if (const auto *mdStr = dyn_cast<MDString>(attr.getOperand(0)))
    return mdStr->getString();
  return "<unknown>";
}

Metadata *llvm::detail::getRawAttrValueMD(const MDNode &attr, unsigned i) {
  // The first operand of the attribute will be the name of the attribute.
  unsigned attrIdx = i + 1;
  if (attrIdx < attr.getNumOperands())
    return attr.getOperand(attrIdx);
  return nullptr;
}

template <typename T>
std::optional<T> llvm::detail::getRawAttrValue(const MDNode &attr, unsigned i) {
  if (Metadata *md = getRawAttrValueMD(attr, i))
    return fromMetadata<T>(md);
  return std::nullopt;
}

template <>
std::optional<MDNode *> llvm::detail::getRawAttrValue(const MDNode &attr,
                                                      unsigned i) {
  if (Metadata *md = getRawAttrValueMD(attr, i))
    if (auto *mdNode = dyn_cast<MDNode>(md))
      return mdNode;
  return std::nullopt;
}

MDNode *llvm::detail::getRawAttr(StringRef attrName, const MDNode *attrList) {
  if (attrList)
    for (Metadata *op : attrList->operands().drop_front())
      if (auto *md = dyn_cast<MDNode>(op))
        if (md->getNumOperands())
          if (auto *mdStr = dyn_cast<MDString>(md->getOperand(0)))
            if (mdStr->getString() == attrName)
              return md;
  return nullptr;
}

iterator_range<detail::AttrIterator>
llvm::detail::getRawAttrsRange(const MDNode *attrList) {
  if (attrList) {
    AttrIterator beg(attrList);
    AttrIterator end(attrList, attrList->getNumOperands());

    return iterator_range(beg, end);
  }
  return iterator_range(AttrIterator(), AttrIterator());
}

MDNode *llvm::detail::getNewAttrList(LLVMContext &ctx) {
  return makeAttrList(ctx, {nullptr});
}

MDNode *llvm::detail::getAttrListWith(StringRef attrName,
                                      ArrayRef<Metadata *> attrVals,
                                      MDNode *attrList, LLVMContext &ctx) {
  MDNode *attr = detail::makeRawAttr(ctx, attrName, attrVals);
  if (std::optional<unsigned> i = getAttrIndex(attrName, attrList))
    return replaceInAttrList(attrList, *i, attr);
  else
    return appendToAttrList(attrList, attr);
}

MDNode *llvm::detail::getAttrListWithout(StringRef attrName, MDNode *attrList) {
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

using SetTTID = SmallSet<TTID, 0>;

using MaybeSetTTID = std::optional<SetTTID>;
using MaybeTTID = std::optional<TTID>;
using MaybeSpawnStrategy = std::optional<TapirSpawnStrategy>;

template Metadata *llvm::detail::makeRawAttrValue(LLVMContext &,
                                                  const int8_t &);
template Metadata *llvm::detail::makeRawAttrValue(LLVMContext &,
                                                  const uint8_t &);
template Metadata *llvm::detail::makeRawAttrValue(LLVMContext &,
                                                  const int16_t &);
template Metadata *llvm::detail::makeRawAttrValue(LLVMContext &,
                                                  const uint16_t &);
template Metadata *llvm::detail::makeRawAttrValue(LLVMContext &,
                                                  const int32_t &);
template Metadata *llvm::detail::makeRawAttrValue(LLVMContext &,
                                                  const uint32_t &);
template Metadata *llvm::detail::makeRawAttrValue(LLVMContext &,
                                                  const int64_t &);
template Metadata *llvm::detail::makeRawAttrValue(LLVMContext &,
                                                  const uint64_t &);
template Metadata *llvm::detail::makeRawAttrValue(LLVMContext &, const float &);
template Metadata *llvm::detail::makeRawAttrValue(LLVMContext &,
                                                  const double &);
template Metadata *llvm::detail::makeRawAttrValue(LLVMContext &,
                                                  const StringRef &);
template Metadata *llvm::detail::makeRawAttrValue(LLVMContext &, const TTID &);
template Metadata *llvm::detail::makeRawAttrValue(LLVMContext &,
                                                  const TapirSpawnStrategy &);
template Metadata *llvm::detail::makeRawAttrValue(LLVMContext &,
                                                  const SetTTID &);

template MaybeI8 llvm::detail::getRawAttrValue(const MDNode &, unsigned);
template MaybeU8 llvm::detail::getRawAttrValue(const MDNode &, unsigned);
template MaybeI16 llvm::detail::getRawAttrValue(const MDNode &, unsigned);
template MaybeU16 llvm::detail::getRawAttrValue(const MDNode &, unsigned);
template MaybeI32 llvm::detail::getRawAttrValue(const MDNode &, unsigned);
template MaybeU32 llvm::detail::getRawAttrValue(const MDNode &, unsigned);
template MaybeI64 llvm::detail::getRawAttrValue(const MDNode &, unsigned);
template MaybeU64 llvm::detail::getRawAttrValue(const MDNode &, unsigned);
template MaybeF32 llvm::detail::getRawAttrValue(const MDNode &, unsigned);
template MaybeF64 llvm::detail::getRawAttrValue(const MDNode &, unsigned);
template MaybeStr llvm::detail::getRawAttrValue(const MDNode &, unsigned);
template MaybeTTID llvm::detail::getRawAttrValue(const MDNode &, unsigned);
template MaybeSpawnStrategy llvm::detail::getRawAttrValue(const MDNode &,
                                                          unsigned);
template MaybeSetTTID llvm::detail::getRawAttrValue(const MDNode &, unsigned);

template bool llvm::detail::verifyRawAttrValueAt(KitVerifier &, const MDNode &,
                                                 unsigned, const MaybeI8 &);
template bool llvm::detail::verifyRawAttrValueAt(KitVerifier &, const MDNode &,
                                                 unsigned, const MaybeU8 &);
template bool llvm::detail::verifyRawAttrValueAt(KitVerifier &, const MDNode &,
                                                 unsigned, const MaybeI16 &);
template bool llvm::detail::verifyRawAttrValueAt(KitVerifier &, const MDNode &,
                                                 unsigned, const MaybeU16 &);
template bool llvm::detail::verifyRawAttrValueAt(KitVerifier &, const MDNode &,
                                                 unsigned, const MaybeI32 &);
template bool llvm::detail::verifyRawAttrValueAt(KitVerifier &, const MDNode &,
                                                 unsigned, const MaybeU32 &);
template bool llvm::detail::verifyRawAttrValueAt(KitVerifier &, const MDNode &,
                                                 unsigned, const MaybeI64 &);
template bool llvm::detail::verifyRawAttrValueAt(KitVerifier &, const MDNode &,
                                                 unsigned, const MaybeU64 &);
template bool llvm::detail::verifyRawAttrValueAt(KitVerifier &, const MDNode &,
                                                 unsigned, const MaybeF32 &);
template bool llvm::detail::verifyRawAttrValueAt(KitVerifier &, const MDNode &,
                                                 unsigned, const MaybeF64 &);
template bool llvm::detail::verifyRawAttrValueAt(KitVerifier &, const MDNode &,
                                                 unsigned, const MaybeStr &);

template bool llvm::detail::verifyRawAttrValueAt(KitVerifier &, const MDNode &,
                                                 unsigned, const MaybeTTID &);
template bool llvm::detail::verifyRawAttrValueAt(KitVerifier &, const MDNode &,
                                                 unsigned,
                                                 const MaybeSpawnStrategy &);
template bool llvm::detail::verifyRawAttrValueAt(KitVerifier &, const MDNode &,
                                                 unsigned,
                                                 const MaybeSetTTID &);
