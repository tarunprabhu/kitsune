//===- IntrinsicUtils.cpp - Utilities for Kitsune's intrinsics ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for Kitsune's intrinsics
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/IntrinsicUtils.h"
#include "kitsune/Core/ConstantUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

KitIntrLowerMode llvm::getKitIntrLowerMode(Intrinsic::ID id) {
  assert(isKitIntrinsic(id) && "Must be a Kitsune intrinsic");
  switch (id) {
#define GET_INTR_LOWERING_SPEC
#define INTR(NAME, LOWER_MODE, ALLOW_PARAM_CAST, ALLOW_RETURN_CAST)            \
  case Intrinsic::NAME: return LOWER_MODE;
#include "kitsune/Core/IntrLibFuncMap.inc"
  }
  llvm_unreachable("getKitIntrLowerMode: Intrinsic ID not handled");
}

bool llvm::isKitIntrinsic(Intrinsic::ID id) {
  return Intrinsic::getBaseName(id).starts_with("llvm.kit.");
}

bool llvm::isKitIntrinsicAsync(Intrinsic::ID id) {
  assert(isKitIntrinsic(id) && "Must be a Kitsune intrinsic");
  return Intrinsic::getBaseName(id).starts_with("llvm.kit.async.");
}

bool llvm::isKitIntrinsicBlocking(Intrinsic::ID id) {
  return not isKitIntrinsicAsync(id);
}

bool llvm::isKitIntrinsicCPU(Intrinsic::ID id) {
  assert(isKitIntrinsic(id) && "Must be a Kitsune intrinsic");
  StringRef baseName = Intrinsic::getBaseName(id);
  return baseName.starts_with("llvm.kit.async.cpu") ||
         baseName.starts_with("llvm.kit.cpu");
}

bool llvm::isKitIntrinsicGPU(Intrinsic::ID id) {
  assert(isKitIntrinsic(id) && "Must be a Kitsune intrinsic");
  StringRef baseName = Intrinsic::getBaseName(id);
  return baseName.starts_with("llvm.kit.async.gpu") ||
         baseName.starts_with("llvm.kit.gpu");
}

bool llvm::isKitIntrinsicUniversal(Intrinsic::ID id) {
  switch (id) {
  case Intrinsic::kit_identity: return true;
  default: return false;
  }
}

Value *llvm::getStreamFromLaunch(const CallBase &call) {
  assert(call.getIntrinsicID() == Intrinsic::kit_async_gpu_kernel_launch &&
         "Instruction must call async_launch_kernel intrinsic");

  // The last parameter of the function type of the callee is the "var arg
  // type". By definition, this is also the argument number of the first
  // variadic argument in the call. The argument immediately before this is the
  // stream.
  Function *callee = call.getCalledFunction();
  FunctionType *calleeTy = callee->getFunctionType();
  return call.getArgOperand(calleeTy->getNumParams() - 1);
}

std::optional<TTID> llvm::getTTIDFromKitIntrCall(const CallBase &call) {
  if (Intrinsic::ID id = call.getIntrinsicID())
    if (isKitIntrinsic(id))
      return fromConstant<TTID>(*cast<Constant>(call.getArgOperand(0)));
  return std::nullopt;
}

Value *llvm::createCall(InsertPosition pt, Intrinsic::ID id,
                        ArrayRef<Value *> args, const Twine &name) {
  BasicBlock *bb = pt.getBasicBlock();
  IRBuilder<> builder(bb, pt);
  return builder.CreateIntrinsic(id, args, /*FMFSource=*/{}, name);
}

Value *llvm::createCall(InsertPosition pt, Intrinsic::ID id,
                        ArrayRef<Type *> overloadTypes, ArrayRef<Value *> args,
                        const Twine &name) {
  BasicBlock *bb = pt.getBasicBlock();
  IRBuilder<> builder(bb, pt);
  return builder.CreateIntrinsic(id, overloadTypes, args, /*FMFSource=*/{},
                                 name);
}
