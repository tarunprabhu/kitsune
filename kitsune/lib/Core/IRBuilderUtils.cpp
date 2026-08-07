//===- IRBuilderUtils.cpp - Utilities for the IRBuilder -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's IRBuilder.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/IRBuilderUtils.h"

using namespace llvm;

Function *llvm::getFunction(IRBuilder<> &builder) {
  if (BasicBlock *bb = builder.GetInsertBlock())
    return bb->getParent();
  return nullptr;
}

Module *llvm::getModule(IRBuilder<> &builder) {
  if (BasicBlock *bb = builder.GetInsertBlock())
    if (Function *f = bb->getParent())
      return f->getParent();
  return nullptr;
}

Value *llvm::createCall(IRBuilder<> &builder, KitFunc libFunc,
                        ArrayRef<Value *> args, StringRef name) {
  Module *m = getModule(builder);
  assert(m && "Builder must be set to a basic block in a module");

  FunctionCallee f = getOrInsertLibFunc(*m, libFunc);
  return builder.CreateCall(f, args, name);
}
