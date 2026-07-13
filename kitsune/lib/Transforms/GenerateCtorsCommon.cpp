//===- GenerateCtorsCommon.cpp - Common code for ctor generation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Code shared by the ctor generators (for CPU-centric and GPU-centric tapir
// targets).
//
//===----------------------------------------------------------------------===//

#include "GenerateCtorsCommon.h"
#include "kitsune/Support/OstreamUtils.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

static std::string getName(StringRef kind, TTID tt) {
  std::string buf;
  raw_string_ostream os(buf);

  os << ".kit." << tt << "." << kind;
  os.flush();

  return buf;
}

static Function *genFunc(Module &m, StringRef name) {
  LLVMContext &ctx = m.getContext();

  Type *ret = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(ret, {}, /*IsVarArg=*/false);
  Function *f = Function::Create(fty, GlobalValue::InternalLinkage, name, &m);

  BasicBlock *bbEntry = BasicBlock::Create(ctx, "entry", f);
  BasicBlock *bbExit = BasicBlock::Create(ctx, "exit", f);

  BranchInst::Create(bbExit, bbEntry);
  ReturnInst::Create(ctx, bbExit);

  return f;
}

llvm::detail::GenerateCtorBase::GenerateCtorBase(TTID tt, const TTOptions &tto)
    : tt(tt), tto(tto) {}

IRBuilder<> llvm::detail::GenerateCtorBase::getBuilderForSkeleton(Function *f) {
  return IRBuilder<>(f->getEntryBlock().getTerminator());
}

Function *llvm::detail::GenerateCtorBase::genCtorSkeleton(Module &m) {
  return genFunc(m, getName("ctor", tt));
}

Function *llvm::detail::GenerateCtorBase::genDtorSkeleton(Module &m) {
  return genFunc(m, getName("dtor", tt));
}
