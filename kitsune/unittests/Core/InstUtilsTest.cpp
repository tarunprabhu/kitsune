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

TEST(KitInstUtils, replaceNonMatchingOperands) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);

  Constant *c0 = ConstantInt::get(i32, 0);
  Constant *c1 = ConstantInt::get(i32, 1);
  Constant *c2 = ConstantInt::get(i32, 2);
  Constant *cf = ConstantFP::get(f32, 0);

  ICmpInst *cmp = new ICmpInst(ICmpInst::ICMP_NE, c0, c0);
  BinaryOperator *binOp = BinaryOperator::Create(Instruction::Add, c0, c1);
  UnaryOperator *unOp = UnaryOperator::Create(Instruction::FNeg, cf);
  ReturnInst *ret = ReturnInst::Create(ctx);

  EXPECT_FALSE(replaceNonMatchingOperands(*cmp, c0, c1));
  EXPECT_EQ(cmp->getOperand(0), c0);
  EXPECT_EQ(cmp->getOperand(1), c0);
  EXPECT_FALSE(replaceNonMatchingOperands(*unOp, cf, c0));
  EXPECT_EQ(unOp->getOperand(0), cf);
  EXPECT_FALSE(replaceNonMatchingOperands(*ret, c0, c2));

  EXPECT_TRUE(replaceNonMatchingOperands(*binOp, c0, c2));
  EXPECT_EQ(binOp->getOperand(0), c0);
  EXPECT_EQ(binOp->getOperand(1), c2);
  EXPECT_TRUE(replaceNonMatchingOperands(*binOp, c2, c1));
  EXPECT_EQ(binOp->getOperand(0), c1);
  EXPECT_EQ(binOp->getOperand(1), c2);
}

TEST(KitInstUtils, replaceMatchingOperands) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);

  Constant *c0 = ConstantInt::get(i32, 0);
  Constant *c1 = ConstantInt::get(i32, 1);
  Constant *c2 = ConstantInt::get(i32, 2);
  Constant *c3 = ConstantInt::get(i32, 3);
  Constant *cf = ConstantFP::get(f32, 0);

  ICmpInst *cmp = new ICmpInst(ICmpInst::ICMP_NE, c0, c0);
  SelectInst *select = SelectInst::Create(cmp, c0, c1);
  UnaryOperator *unOp = UnaryOperator::Create(Instruction::FNeg, cf);
  ReturnInst *ret = ReturnInst::Create(ctx, c3);
  ReturnInst *retVoid = ReturnInst::Create(ctx);

  EXPECT_FALSE(replaceMatchingOperands(*cmp, c1, c2));
  EXPECT_FALSE(replaceMatchingOperands(*select, c2, c3));
  EXPECT_FALSE(replaceMatchingOperands(*unOp, c3, c0));
  EXPECT_EQ(unOp->getOperand(0), cf);
  EXPECT_FALSE(replaceMatchingOperands(*ret, c0, c1));
  EXPECT_EQ(ret->getOperand(0), c3);
  EXPECT_FALSE(replaceMatchingOperands(*retVoid, cmp, c0));

  // If the operand was replaced with itself, this will return false.
  EXPECT_FALSE(replaceMatchingOperands(*cmp, c0, c0));
  EXPECT_EQ(cmp->getOperand(0), c0);
  EXPECT_EQ(cmp->getOperand(1), c0);
  EXPECT_TRUE(replaceMatchingOperands(*cmp, c0, c1));
  EXPECT_EQ(cmp->getOperand(0), c1);
  EXPECT_EQ(cmp->getOperand(1), c1);
  EXPECT_TRUE(replaceMatchingOperands(*ret, c3, c0));
  EXPECT_EQ(ret->getOperand(0), c0);
  EXPECT_TRUE(replaceMatchingOperands(*select, c0, c3));
  EXPECT_EQ(select->getOperand(1), c3);
  EXPECT_EQ(select->getOperand(2), c1);
  EXPECT_TRUE(replaceMatchingOperands(*select, c1, c0));
  EXPECT_EQ(select->getOperand(1), c3);
  EXPECT_EQ(select->getOperand(2), c0);
}

TEST(KitInstUtils, getNonMatchingOperand) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);

  Constant *c0 = ConstantInt::get(i32, 0);
  Constant *c1 = ConstantInt::get(i32, 1);
  Constant *c2 = ConstantInt::get(i32, 2);
  Constant *cf = ConstantFP::get(f32, 0);

  BinaryOperator *iop = BinaryOperator::Create(Instruction::Add, c0, c1);
  BinaryOperator *fop = BinaryOperator::Create(Instruction::FMul, cf, cf);

  EXPECT_EQ(getNonMatchingOperand(*iop, c0), c1);
  EXPECT_EQ(getNonMatchingOperand(*iop, c1), c0);
  EXPECT_FALSE(getNonMatchingOperand(*iop, c2));
  EXPECT_FALSE(getNonMatchingOperand(*iop, cf));
  EXPECT_FALSE(getNonMatchingOperand(*fop, cf));
}

TEST(KitInstUtils, getNonMatchingSuccessor) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);

  BasicBlock *def = BasicBlock::Create(ctx);
  BasicBlock *bt = BasicBlock::Create(ctx);
  BasicBlock *bf = BasicBlock::Create(ctx);

  ConstantInt *c1 = cast<ConstantInt>(ConstantInt::get(i32, 1));
  ConstantInt *c2 = cast<ConstantInt>(ConstantInt::get(i32, 2));
  ConstantInt *c3 = cast<ConstantInt>(ConstantInt::get(i32, 3));

  BinaryOperator *binOp = BinaryOperator::Create(Instruction::Add, c1, c2);
  BranchInst *uncondBr = BranchInst::Create(def);
  BranchInst *condBr = BranchInst::Create(bt, bf, ConstantInt::getTrue(ctx));
  SwitchInst *swch = SwitchInst::Create(c3, def, 2);
  swch->addCase(c1, bt);
  swch->addCase(c2, bf);

  EXPECT_FALSE(getNonMatchingSuccessor(*binOp, def));
  EXPECT_FALSE(getNonMatchingSuccessor(*uncondBr, def));
  EXPECT_FALSE(getNonMatchingSuccessor(*swch, def));
  EXPECT_FALSE(getNonMatchingSuccessor(*swch, bf));
  EXPECT_FALSE(getNonMatchingSuccessor(*swch, bt));
  EXPECT_EQ(getNonMatchingSuccessor(*condBr, bt), bf);
  EXPECT_EQ(getNonMatchingSuccessor(*condBr, bf), bt);
}

TEST(KitInstUtils, isCondBr) {
  LLVMContext ctx;
  BasicBlock *bt = BasicBlock::Create(ctx);
  BasicBlock *bf = BasicBlock::Create(ctx);
  Constant *cond = ConstantInt::getTrue(ctx);
  BranchInst *condBr = BranchInst::Create(bt, bf, cond);
  BranchInst *uncondBr = BranchInst::Create(bt);

  EXPECT_TRUE(isCondBr(*condBr));
  EXPECT_FALSE(isCondBr(*uncondBr));
}

TEST(KitInstUtils, isUncondBr) {
  LLVMContext ctx;
  BasicBlock *bt = BasicBlock::Create(ctx);
  BasicBlock *bf = BasicBlock::Create(ctx);
  Constant *cond = ConstantInt::getTrue(ctx);
  BranchInst *condBr = BranchInst::Create(bt, bf, cond);
  BranchInst *uncondBr = BranchInst::Create(bt);

  EXPECT_TRUE(isUncondBr(*uncondBr));
  EXPECT_FALSE(isUncondBr(*condBr));
}

} // namespace
