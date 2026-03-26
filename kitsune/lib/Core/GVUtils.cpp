//===- GVUtils.cpp - Utilities for LLVM GlobalVariables -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM GlobalVariable's.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/GVUtils.h"
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;

LLVMContext &llvm::getContext(const GlobalVariable &g) {
  return g.getContext();
}
