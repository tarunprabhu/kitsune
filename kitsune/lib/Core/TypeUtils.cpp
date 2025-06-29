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

bool llvm::isByteArrayTy(Type *Ty) {
  if (auto *arrayTy = dyn_cast<ArrayType>(Ty))
    return arrayTy->getElementType()->isIntegerTy(8);
  return false;
}
