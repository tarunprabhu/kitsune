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
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/TTUtils.h"

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

static CallInst *createKitMalloc(IRBuilder<> &builder, Intrinsic::ID intr,
                                 TTID tt, Value *bytes, StringRef name) {
  LLVMContext &ctx = builder.getContext();
  Constant *ctt = toConstant(tt, ctx);

  CallInst *call = cast<CallInst>(builder.CreateIntrinsic(intr, {ctt, bytes}));
  call->setAttributes(AttributeList().addRetAttribute(ctx, Attribute::NoAlias));
  call->setTailCall();

  return call;
}

CallInst *llvm::createCPUMalloc(IRBuilder<> &builder, TTID tt, Value *bytes,
                                StringRef name) {
  assert(!isGPUTT(tt) && "Tapir target must be CPU-centric");
  return createKitMalloc(builder, Intrinsic::kit_cpu_malloc, tt, bytes, name);
}

CallInst *llvm::createGPUMalloc(IRBuilder<> &builder, TTID tt, Value *bytes,
                                StringRef name) {
  assert(isGPUTT(tt) && "Tapir target must be GPU-centric");
  return createKitMalloc(builder, Intrinsic::kit_gpu_malloc, tt, bytes, name);
}
