//===- TypeUtils.cpp - Helper functions for types -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of utilities for types
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TypeUtils.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

using namespace llvm;

bool llvm::isByteArrayTy(Type *ty) {
  if (auto *aty = dyn_cast<ArrayType>(ty))
    return aty->getElementType()->isIntegerTy(8);
  return false;
}

bool llvm::isZeroLenByteArrayTy(Type *ty) {
  if (auto *aty = dyn_cast<ArrayType>(ty))
    return aty->getElementType()->isIntegerTy(8) and aty->getNumElements() == 0;
  return false;
}
