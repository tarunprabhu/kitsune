//===- GlobalVariableUtils.cpp - Utilities for global variables -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of utilities to deal with global variables
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/GlobalVariableUtils.h"
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;

std::optional<TTID> llvm::getAttrValueAsTTID(const GlobalVariable &g,
                                             Attribute::AttrKind attr) {
  if (g.hasAttribute(attr))
    return g.getAttribute(attr).getTTID();
  return std::nullopt;
}
