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

TEST(KitBasicBlockUtils, isDeadEnd) {
  LLVMContext ctx;

  // Empty basic blocks are not dead-ends.
  BasicBlock *bbU = BasicBlock::Create(ctx);
  EXPECT_FALSE(isDeadEnd(*bbU));

  // Terminator of the basic block is an unconditional branch, but the sole
  // successor of the block is not a dead-end.
  //
  //   bb1 -> bbU
  //
  BasicBlock *bb1 = BasicBlock::Create(ctx);
  BranchInst::Create(bbU, bb1);
  EXPECT_FALSE(isDeadEnd(*bb1));

  // Basic block now contains an unreachable instruction. It is a dead-end.
  (void)new UnreachableInst(ctx, bbU->getFirstInsertionPt());
  EXPECT_TRUE(isDeadEnd(*bbU));

  // Terminator of the basic block is an unconditional branch. The sole
  // successor of the block is a dead-end.
  EXPECT_TRUE(isDeadEnd(*bb1));

  // Traverse multiple unconditional branches to determine if it is a dead-end.
  //
  //   bb2 -> bb3 -> bbU
  //
  BasicBlock *bb2 = BasicBlock::Create(ctx);
  BasicBlock *bb3 = BasicBlock::Create(ctx);
  BranchInst::Create(bb3, bb2);
  BranchInst::Create(bbU, bb3);
  EXPECT_TRUE(isDeadEnd(*bb2));
  EXPECT_TRUE(isDeadEnd(*bb3));

  // Even if both branches of a conditional branch are dead-ends, the block
  // itself will not be a dead-end.
  Constant *cond = ConstantInt::getTrue(ctx);
  BasicBlock *bb4 = BasicBlock::Create(ctx);
  BranchInst::Create(bbU, bbU, cond, bb4);

  EXPECT_FALSE(isDeadEnd(*bb4));
}

} // namespace
