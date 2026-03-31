//===- AttrsIterator.h - Iterator over raw attribute lists ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// iterator over raw attribute lists.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_CORE_ATTRS_ITERATOR_H
#define KITSUNE_LIB_CORE_ATTRS_ITERATOR_H

#include "llvm/IR/Metadata.h"

namespace llvm {

namespace detail {

/// Iterator over a raw attribute list.
class AttrIterator {
public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = MDNode &;
  using pointer = MDNode *;
  using reference = MDNode &;

public:
  AttrIterator() : attrList(nullptr), curr(0) {}
  AttrIterator(const MDNode *attrList) : attrList(attrList), curr(1) {}
  AttrIterator(const MDNode *attrList, unsigned last)
      : attrList(attrList), curr(last) {}

  reference operator*() const {
    return *cast<MDNode>(attrList->getOperand(curr));
  }

  pointer operator->() const {
    return cast<MDNode>(attrList->getOperand(curr));
  }

  AttrIterator &operator++() {
    curr++;
    return *this;
  }

  AttrIterator operator++(int) {
    AttrIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  friend bool operator==(const AttrIterator &l, const AttrIterator &r) {
    return l.attrList == r.attrList && l.curr == r.curr;
  }

  friend bool operator!=(const AttrIterator &l, const AttrIterator &r) {
    return !(l == r);
  }

private:
  const MDNode *attrList = nullptr;
  unsigned curr = 0;
};

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_CORE_ATTRS_ITERATOR_H
