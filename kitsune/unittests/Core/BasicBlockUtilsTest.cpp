//==- BasicBlockUtilsTest.cpp - Unit tests for Kitsune's basic block utils -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/BasicBlockUtils.h"
#include "TestUtils.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

static constexpr StringRef ll = R"(
define i64 @f(i64 %0) {
  br label %end

end:
  ret i64 %0
}
)";

namespace {

TEST(KitBasicBlockUtils, getModule) {
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

  EXPECT_FALSE(getModule(*bbO));
  EXPECT_FALSE(getModule(*bbFO));
  EXPECT_EQ(getModule(*bbFM), &m);
}

TEST(KitBasicBlockUtils, getName) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);

  Function *f = m->getFunction("f");
  SmallVector<std::string> names;
  for (BasicBlock &bb : *f)
    names.push_back(getName(bb));

  unsigned i = 0;
  EXPECT_EQ(names[i++], "%1");  // The entry basic block is unnamed
  EXPECT_EQ(names[i++], "end"); // The basic block is named
}

TEST(KitBasicBlockUtils, isDisconnected) {
  LLVMContext ctx;
  BasicBlock *bb0 = BasicBlock::Create(ctx);
  BasicBlock *bbNoPreds = BasicBlock::Create(ctx);
  BasicBlock *bbNoSuccs = BasicBlock::Create(ctx);

  BranchInst::Create(bbNoSuccs, bbNoPreds->getFirstInsertionPt());
  ReturnInst::Create(ctx, bbNoSuccs);

  // The basic block has neither predecessors, nor successors.
  EXPECT_TRUE(isDisconnected(*bb0));

  // Basic block has no predecessors, but does have a successor, and is,
  // therefore, not disconnected.
  EXPECT_FALSE(isDisconnected(*bbNoPreds));

  // The basic block has no successors, but does have a predecessor, and is,
  // therefore, not disconnected.
  EXPECT_FALSE(isDisconnected(*bbNoSuccs));
}

TEST(KitBasicBlockUtils, isOrphaned) {
  LLVMContext ctx;
  BasicBlock *bb0 = BasicBlock::Create(ctx);
  BasicBlock *bbNoPreds = BasicBlock::Create(ctx);
  BasicBlock *bbNoSuccs = BasicBlock::Create(ctx);

  BranchInst::Create(bbNoSuccs, bbNoPreds->getFirstInsertionPt());
  ReturnInst::Create(ctx, bbNoSuccs);

  // The basic block has neither predecessors, nor successors. It is, therefore,
  // orphaned.
  EXPECT_TRUE(isOrphaned(*bb0));

  // Basic block has no predecessors. It does have a successor, but even so, it
  // is orphaned.
  EXPECT_TRUE(isOrphaned(*bbNoPreds));

  // The basic block has no successors, but does have a predecessor, and is,
  // therefore, not orphaned.
  EXPECT_FALSE(isOrphaned(*bbNoSuccs));
}

TEST(KitBasicBlockUtils, isUnreachable) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);

  BasicBlock *bbEmpty = BasicBlock::Create(ctx);

  Constant *c0 = ConstantInt::get(i32, 0);
  Constant *c1 = ConstantInt::get(i32, 1);

  // Empty basic blocks cannot be unreachable.
  EXPECT_FALSE(isUnreachable(*bbEmpty));

  // Basic block has a single instruction, but does not contain an unreachable
  // instruction.
  BasicBlock *bbBranches = BasicBlock::Create(ctx);
  BranchInst::Create(bbEmpty, bbBranches);
  EXPECT_FALSE(isUnreachable(*bbBranches));

  // Basic block contains an unreachable instruction, but it is not the only
  // instruction in the basic block.
  BasicBlock *bbNotU = BasicBlock::Create(ctx);
  (void)BinaryOperator::Create(Instruction::Add, c0, c1, "",
                               bbNotU->getFirstInsertionPt());
  (void)new UnreachableInst(ctx, bbNotU->end());
  EXPECT_FALSE(isUnreachable(*bbNotU));

  // Basic block contains nothing but an unreachable instruction. It is
  // unreachable.
  BasicBlock *bbU = BasicBlock::Create(ctx);
  (void)new UnreachableInst(ctx, bbU->getFirstInsertionPt());
  EXPECT_TRUE(isUnreachable(*bbU));
}

} // namespace
