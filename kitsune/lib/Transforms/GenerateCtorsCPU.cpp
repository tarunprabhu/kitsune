//===- GenerateCtorsCPU.cpp - Generate ctors for CPU tapir targets --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic implementation of ctor/dtor generation for CPU-centric tapir targets.
// This is usually sufficient for most of these tapir targets.
//
//===----------------------------------------------------------------------===//

#include "GenerateCtorsCPU.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/TTOptions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

detail::GenerateCtorCPU::GenerateCtorCPU(TTID tt, const TTOptions &tto)
    : GenerateCtorBase(tt, tto) {}

Function *detail::GenerateCtorCPU::genCtor(Module &m) {
  LLVMContext &ctx = m.getContext();

  Constant *ctt = toConstant(tt, ctx);

  Function *ctor = genCtorSkeleton(m);
  IRBuilder<> builder = getBuilderForSkeleton(ctor);

  builder.CreateIntrinsic(Intrinsic::kit_runtime_initialize, ctt);

  // We don't need to do anything more because genCtorSkeleton() will have set
  // up dedicated exit blocks and return instructions already.
  return ctor;
}

Function *detail::GenerateCtorCPU::genDtor(Module &m) {
  LLVMContext &ctx = m.getContext();

  Constant *ctt = toConstant(tt, ctx);

  Function *dtor = genDtorSkeleton(m);
  IRBuilder<> builder = getBuilderForSkeleton(dtor);

  builder.CreateIntrinsic(Intrinsic::kit_runtime_finalize, {ctt});

  // We don't need to do anything more because genCtorSkeleton() will have set
  // up dedicated exit blocks and return instructions already.
  return dtor;
}

void detail::GenerateCtorCPU::run(Module &m) {
  appendToGlobalCtors(m, genCtor(m), kitCtorPriority);
  appendToGlobalDtors(m, genDtor(m), kitDtorPriority);
}
