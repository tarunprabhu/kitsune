//- GenerateCtorOpenCilk.cpp - Generate ctor for Kitsune's OpenCilk runtime --//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generate a global constructor and destructor for Kitsune's OpenCilk runtime.
//
//===----------------------------------------------------------------------===//

#include "GenerateCtorsImpl.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace {

/// Helper class to generate a ctor for kitocilk (Kitsune's runtime for the
/// opencilk tapir target).
class GenerateCtorOpenCilk {
private:
  detail::GetTLI getTLI;
  const TTOptions &tto;

private:
  Function *createCtor(Module &m, Function *dtor);
  Function *createDtor(Module &m);

public:
  GenerateCtorOpenCilk(detail::GetTLI getTLI, const TTOptions &tto);

  void run(Module &m);
};

} // namespace

GenerateCtorOpenCilk::GenerateCtorOpenCilk(detail::GetTLI getTLI,
                                           const TTOptions &tto)
    : getTLI(getTLI), tto(tto) {}

Function *GenerateCtorOpenCilk::createCtor(Module &m, Function *dtor) {
  LLVMContext &ctx = m.getContext();
  IRBuilder<> builder(ctx);

  Type *voidTy = Type::getVoidTy(ctx);

  // Booleans are always 8-bit integers. toConstant would, otherwise return
  // an i1, but the intrinsic expects i8. Casting the boolean to i8 ensures
  // that we get a value of the correct type.
  Constant *verbose = toConstant(uint8_t(tto.getKitrtVerbose()), ctx);
  Constant *tt = toConstant(TTID::OpenCilk, ctx);

  FunctionType *ctorTy = FunctionType::get(voidTy, {}, /*IsVarArg=*/false);
  Function *ctor = Function::Create(ctorTy, GlobalValue::InternalLinkage,
                                    ".kitocilk.ctor", &m);

  BasicBlock *bbEntry = BasicBlock::Create(ctx, "entry", ctor);
  BasicBlock *bbExit = BasicBlock::Create(ctx, "exit", ctor);

  builder.SetInsertPoint(bbEntry);

  // We can't enable verbose mode until after we call initialize.
  builder.CreateIntrinsic(Intrinsic::kit_runtime_initialize, tt);
  builder.CreateIntrinsic(Intrinsic::kit_runtime_set_verbose, {tt, verbose});

  // Now add the dtor to help us clean up at program exit.
  TargetLibraryInfo &tli = getTLI(*ctor);
  FunctionCallee atExit = getOrInsertLibFunc(&m, tli, LibFunc_atexit);
  builder.CreateCall(atExit, dtor);

  builder.CreateBr(bbExit);

  builder.SetInsertPoint(bbExit);
  builder.CreateRetVoid();

  return ctor;
}

Function *GenerateCtorOpenCilk::createDtor(Module &m) {
  LLVMContext &ctx = m.getContext();
  IRBuilder<> builder(ctx);

  Type *voidTy = Type::getVoidTy(ctx);

  Constant *ctt = toConstant(TTID::OpenCilk, ctx);

  FunctionType *dtorTy = FunctionType::get(voidTy, {}, /*IsVarArg=*/false);
  Function *dtor = Function::Create(dtorTy, GlobalValue::InternalLinkage,
                                    ".kitocilk.dtor", &m);

  BasicBlock *bbEntry = BasicBlock::Create(ctx, "entry", dtor);
  BasicBlock *bbExit = BasicBlock::Create(ctx, "exit", dtor);

  builder.SetInsertPoint(bbEntry);
  builder.CreateIntrinsic(Intrinsic::kit_runtime_finalize, {ctt});
  builder.CreateBr(bbExit);

  builder.SetInsertPoint(bbExit);
  builder.CreateRetVoid();

  return dtor;
}

void GenerateCtorOpenCilk::run(Module &m) {
  Function *dtor = createDtor(m);
  Function *ctor = createCtor(m, dtor);

  // The priority must be in the range [101,65535] with larger values having
  // lower priority relative to other global constructors in @llvm.global_ctors.
  appendToGlobalCtors(m, ctor, 65535);
}

void llvm::detail::genCtorOpenCilk(Module &m, detail::GetTLI getTLI,
                                   const TTOptions &tto,
                                   const detail::GenerateCtorOptions &) {
  GenerateCtorOpenCilk(getTLI, tto).run(m);
}
