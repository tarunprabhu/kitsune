//===- FunctionUtils.cpp - Utilities for LLVM functions -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM Function's.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/FunctionUtils.h"
#include "llvm/IR/Function.h"

using namespace llvm;

void llvm::copyAttrs(Function &dst, Function &src) {
  for (Attribute attr : src.getAttributes().getFnAttrs())
    dst.addAttributeAtIndex(AttributeList::FunctionIndex, attr);

  for (Attribute attr : src.getAttributes().getRetAttrs())
    dst.addAttributeAtIndex(AttributeList::ReturnIndex, attr);

  dst.setCallingConv(src.getCallingConv());
  if (src.hasGC())
    dst.setGC(src.getGC());
  else
    dst.clearGC();
  if (src.hasPersonalityFn())
    dst.setPersonalityFn(src.getPersonalityFn());
  if (src.hasPrefixData())
    dst.setPrefixData(src.getPrefixData());
  if (src.hasPrologueData())
    dst.setPrologueData(src.getPrologueData());
}

void llvm::copyAttrs(Argument &dst, Argument &src) {
  for (Attribute attr : src.getAttributes())
    dst.addAttr(attr);
}
