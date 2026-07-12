//=- GenerateCtorQthreads.cpp - Generate ctor for Kitsune's qthreads runtime =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generate a global constructor and destructor for Kitsune's qthreads runtime.
//
//===----------------------------------------------------------------------===//

#include "GenerateCtorsImpl.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace {

/// Helper class to generate a ctor for kitqthr (Kitsune's runtime for the
/// qthreads tapir target).
class GenerateCtorQthreads {
private:
  detail::GetTLI getTLI;
  const TTOptions &tto;

private:
  void genCtor(Module &m);
  void genDtor(Module &m);

public:
  GenerateCtorQthreads(detail::GetTLI getTLI, const TTOptions &tto);

  void run(Module &m);
};

} // namespace

GenerateCtorQthreads::GenerateCtorQthreads(detail::GetTLI getTLI,
                                           const TTOptions &tto)
    : getTLI(getTLI), tto(tto) {}

void GenerateCtorQthreads::genCtor(Module &m) {
  LLVMContext &ctx = m.getContext();

  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *ctorTy = FunctionType::get(voidTy, {}, false);

  // Booleans are always 8-bit integers. toConstant would, otherwise return an
  // i1, but the intrinsic expects i8. Casting the boolean to i8 ensures that we
  // get a value of the correct type.
  Constant *verbose = toConstant(uint8_t(tto.getKitrtVerbose()), ctx);
  Constant *tt = toConstant(TTID::Qthreads, ctx);

  Function *ctor = Function::Create(ctorTy, GlobalValue::InternalLinkage,
                                    ".kitqthr.ctor", &m);
  BasicBlock *bbEntry = BasicBlock::Create(ctx, "entry", ctor);
  BasicBlock *bbExit = BasicBlock::Create(ctx, "exit", ctor);
  IRBuilder<> builder(ctx);

  builder.SetInsertPoint(bbEntry);
  builder.CreateIntrinsic(Intrinsic::kit_runtime_initialize, {tt});
  builder.CreateIntrinsic(Intrinsic::kit_runtime_set_verbose, {tt, verbose});
  builder.CreateBr(bbExit);

  builder.SetInsertPoint(bbExit);
  builder.CreateRetVoid();

  appendToGlobalCtors(m, ctor, detail::kitCtorPriority);
}

void GenerateCtorQthreads::genDtor(Module &m) {
  LLVMContext &ctx = m.getContext();

  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *dtorTy = FunctionType::get(voidTy, {}, false);

  Constant *tt = toConstant(TTID::Qthreads, ctx);

  Function *dtor = Function::Create(dtorTy, GlobalValue::InternalLinkage,
                                    ".kitqthr.dtor", &m);
  BasicBlock *bbEntry = BasicBlock::Create(ctx, "entry", dtor);
  BasicBlock *bbExit = BasicBlock::Create(ctx, "exit", dtor);
  IRBuilder<> builder(ctx);

  builder.SetInsertPoint(bbEntry);
  builder.CreateIntrinsic(Intrinsic::kit_runtime_finalize, {tt});
  builder.CreateBr(bbExit);

  builder.SetInsertPoint(bbExit);
  builder.CreateRetVoid();

  appendToGlobalDtors(m, dtor, detail::kitDtorPriority);
}

void GenerateCtorQthreads::run(Module &m) {
  genCtor(m);
  genDtor(m);
}

void llvm::detail::genCtorQthreads(Module &m, detail::GetTLI getTLI,
                                   const TTOptions &tto,
                                   const detail::GenerateCtorOptions &) {
  GenerateCtorQthreads(getTLI, tto).run(m);
}
