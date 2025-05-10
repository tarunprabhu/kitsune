//===- KitsuneUtils.cpp - Helper functions for Kitsune ---------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions for Kitsune-specific utilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/KitsuneUtils.h"

using namespace llvm;

ConstantInt *llvm::getConstantInt(LLVMContext &ctxt, TapirTargetID tt) {
  IntegerType *i8 = IntegerType::get(ctxt, 8);
  return ConstantInt::get(i8, int(tt), false);
}
