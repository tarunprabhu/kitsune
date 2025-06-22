//===- KernelProperties.cpp - Kernel function properties ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of functions to help with calculating and saving the properties
// of kernel functions that are used by Kitsune's runtime.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/KernelProperties.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace llvm;

StructType *llvm::getKernelInstMixType(LLVMContext &ctx) {
  Type *i64 = Type::getInt64Ty(ctx);
  return StructType::get(i64,  // number of memory ops
                         i64,  // number of floating point ops
                         i64,  // number of integer ops
                         i64); // number of other ops
}

GlobalVariable *llvm::createKernelPropsGlobal(StringRef kernelName, Module &m) {
  LLVMContext &ctx = m.getContext();
  StructType *type = getKernelInstMixType(ctx);
  Constant *init = Constant::getNullValue(type);
  auto *g = new GlobalVariable(m, type, /*IsConstant=*/true,
                               GlobalValue::PrivateLinkage, init);

  g->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  g->addAttribute(Attribute::getWithKernelProps(ctx, kernelName));

  return g;
}
