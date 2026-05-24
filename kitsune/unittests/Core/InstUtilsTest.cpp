//===- InstUtilsTest.cpp - Unit tests for Kitsune's instruction utilities -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/InstUtils.h"
#include "TestUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

static constexpr StringRef ll = R"(
declare void @ext()
declare ptr @get()

define i64 @f(i64 %0) {
  call void @ext()
  %2 = call ptr @get()
  call void %2()
  br label %end

end:
  ret i64 %0
}
)";

namespace {

TEST(KitInstUtils, getModule) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *funcTy = FunctionType::get(voidTy, {}, /*IsVarArg=*/false);
  GlobalValue::LinkageTypes linkage = GlobalValue::ExternalLinkage;

  Module m("", ctx);
  Function *fO = Function::Create(funcTy, linkage, "fO");
  Function *fM = Function::Create(funcTy, linkage, "fM", &m);

  BasicBlock *bbO = BasicBlock::Create(ctx);
  BasicBlock *bbFO = BasicBlock::Create(ctx, "", fO);
  BasicBlock *bbFM = BasicBlock::Create(ctx, "", fM);

  ReturnInst *retO = ReturnInst::Create(ctx);
  ReturnInst *retBBO = ReturnInst::Create(ctx, bbO);
  ReturnInst *retFO = ReturnInst::Create(ctx, bbFO);
  ReturnInst *retFM = ReturnInst::Create(ctx, bbFM);

  EXPECT_FALSE(getModule(*retO));
  EXPECT_FALSE(getModule(*retBBO));
  EXPECT_FALSE(getModule(*retFO));
  EXPECT_EQ(getModule(*retFM), &m);
}

TEST(KitInstUtils, getName) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);

  SmallVector<std::string, 4> names;
  Function *f = m->getFunction("f");
  for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
    names.push_back(getName(*i));

  unsigned i = 0;
  EXPECT_EQ(names[i++], "<call ext>");      // Call does not return a value
  EXPECT_EQ(names[i++], "%2");              // The call returns a value
  EXPECT_EQ(names[i++], "<call %2>");       // The instruction is named
  EXPECT_EQ(names[i++], "<br label %end>"); // Branches have no name
  EXPECT_EQ(names[i++], "<ret i64 %0>");    // Returns have no name
}

TEST(KitInstUtils, getInstClassName) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f64 = Type::getDoubleTy(ctx);
  PointerType *ptr = PointerType::getUnqual(ctx);
  FunctionType *fty = FunctionType::get(i32, i64, /*isVarArg=*/false);

  ConstantPointerNull *cnull = ConstantPointerNull::get(ptr);
  Constant *c0 = ConstantInt::get(i64, 0);
  Constant *c1 = ConstantInt::get(i64, 1);
  Constant *cf = ConstantFP::get(f64, 0.0);

  // These work just fine without having to add them to a function. Using the
  // builder only complicates things because it will fold constants - which is
  // very annoying.
  BinaryOperator *binOp = BinaryOperator::Create(Instruction::Add, c0, c1);
  UnaryOperator *unOp = UnaryOperator::Create(Instruction::FNeg, cf);
  ReturnInst *ret = ReturnInst::Create(ctx);
  PHINode *phi = PHINode::Create(i64, 0);

  // However, the load and trunc instructions don't seem to work if you try to
  // create them outside a function, so go through the whole rigmarole of
  // setting up an IR builder for them.
  Module m("", ctx);
  Function *f = cast<Function>(m.getOrInsertFunction("f", fty).getCallee());
  Argument *arg = f->getArg(0);
  BasicBlock *bb = BasicBlock::Create(ctx, "entry", f);
  IRBuilder<> builder(bb);

  LoadInst *load = builder.CreateLoad(i64, cnull);
  TruncInst *trunc = cast<TruncInst>(builder.CreateTruncOrBitCast(arg, i32));

  EXPECT_EQ(getInstClassName(*ret), "ReturnInst");
  EXPECT_EQ(getInstClassName(*binOp), "BinaryOperator");
  EXPECT_EQ(getInstClassName(*unOp), "UnaryOperator");
  EXPECT_EQ(getInstClassName(*load), "LoadInst");
  EXPECT_EQ(getInstClassName(*trunc), "TruncInst");
  EXPECT_EQ(getInstClassName(*phi), "PHINode");
}

} // namespace
