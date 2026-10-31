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
                        ArrayRef<Value *> args, StringRef name,
                        bool allowCasts) {
  Module *m = getModule(builder);
  assert(m && "Builder must be set to a basic block in a module");

  FunctionCallee f = getOrInsertLibFunc(*m, libFunc);
  if (!allowCasts || !isa<Function>(f.getCallee()))
    return builder.CreateCall(f, args, name);

  // Since casts are allowed, if the types of any arguments don't match what the
  // callee expects, cast this.
  Function *callee = cast<Function>(f.getCallee());
  FunctionType *calleeTy = callee->getFunctionType();
  SmallVector<Value *, 4> cstArgs;
  for (unsigned i = 0; i < args.size(); ++i) {
    Value *arg = args[i];
    Type *ty = calleeTy->getParamType(i);
    if (ty->isIntegerTy())
      cstArgs.push_back(builder.CreateIntCast(arg, ty, /*isSigned=*/true));
    else if (ty->isPointerTy())
      cstArgs.push_back(builder.CreatePointerBitCastOrAddrSpaceCast(arg, ty));
    else
      cstArgs.push_back(arg);
  }
  return builder.CreateCall(f, cstArgs, name);
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
