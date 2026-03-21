//===- LoopAttrsTest.cpp - Unit tests for Kitsune's loop attributes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/LoopAttrs.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

static constexpr StringRef ll = R"(
define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp.not = icmp eq i64 %n, 0
  br i1 %cmp.not, label %for.i.sync, label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %exitcond.i.not = icmp eq i64 %inc.i, %n
  br i1 %exitcond.i.not, label %for.i.sync, label %for.i.header, !llvm.loop !0

for.i.sync:
  sync within %syncreg, label %for.i.exit

for.i.exit:
  ret void
}

!0 = distinct !{!0}
)";

static std::unique_ptr<Module> parseIR(LLVMContext &ctx, StringRef ir) {
  SMDiagnostic err;
  std::unique_ptr<Module> m = parseAssemblyString(ir, err, ctx);
  if (!m)
    err.print("parseIR", errs());
  return m;
}

TEST(KitLoopAttrs, loopGetMetadata) {
#define CHECK_METADATA_0(ATTR)                                                 \
  {                                                                            \
    LLVMContext ctx;                                                           \
    MDNode *md = getMDNodeForAttr(ctx, ATTR);                                  \
    EXPECT_EQ(md->getNumOperands(), 1U);                                       \
                                                                               \
    auto *md0 = dyn_cast<MDString>(md->getOperand(0));                         \
    EXPECT_TRUE(md0);                                                          \
    EXPECT_EQ(md0->getString(), getAttrName(ATTR));                            \
  }

  CHECK_METADATA_0(LoopAttrKind::LoweringEnabled)

#define CHECK_METADATA_1(ATTR, VAL, EXP)                                       \
  {                                                                            \
    LLVMContext ctx;                                                           \
    MDNode *md = getMDNodeForAttr(ctx, ATTR, VAL);                             \
    EXPECT_EQ(md->getNumOperands(), 2U);                                       \
                                                                               \
    auto *md0 = dyn_cast<MDString>(md->getOperand(0));                         \
    EXPECT_TRUE(md0);                                                          \
    EXPECT_EQ(md0->getString(), getAttrName(ATTR));                            \
                                                                               \
    auto *md1 = dyn_cast<ConstantAsMetadata>(md->getOperand(1));               \
    EXPECT_TRUE(md1);                                                          \
    EXPECT_EQ(cast<ConstantInt>(md1->getValue())->getLimitedValue(), EXP);     \
  }

  CHECK_METADATA_1(LoopAttrKind::Target, TTID::Serial, 1U);
  CHECK_METADATA_1(LoopAttrKind::PerfectDepth, 13, 13U);
}

TEST(KitLoopAttrs, attrName) {
#define LOOP_ATTR(NAME, TYPE, TAPIRONLY, IRNAME)                               \
  EXPECT_EQ(getAttrName(LoopAttrKind::NAME), IRNAME);                          \
  if constexpr (TAPIRONLY)                                                     \
    EXPECT_TRUE(getAttrName(LoopAttrKind::NAME).starts_with("tapir.loop."));   \
  else                                                                         \
    EXPECT_TRUE(getAttrName(LoopAttrKind::NAME).starts_with("loop."));
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attrKind) {
  EXPECT_EQ(getLoopAttrKind("whoops"), std::nullopt);

#define LOOP_ATTR(NAME, TYPE, TAPIRONLY, IRNAME)                               \
  EXPECT_EQ(getLoopAttrKind(IRNAME), LoopAttrKind::NAME);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attrTapirOnly) {
#define LOOP_ATTR(NAME, TYPE, TAPIRONLY, IRNAME)                               \
  EXPECT_EQ(isAttrTapirOnly(LoopAttrKind::NAME), TAPIRONLY);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

#define CHECK_GENERIC(NAME, LOOP, VAL)                                         \
  EXPECT_FALSE(hasAttr(LOOP, LoopAttrKind::NAME));                             \
  add##NAME##Attr(LOOP, VAL);                                                  \
  EXPECT_TRUE(hasAttr(LOOP, LoopAttrKind::NAME));                              \
  removeAttr(LOOP, LoopAttrKind::NAME);                                        \
  EXPECT_FALSE(hasAttr(LOOP, LoopAttrKind::NAME));                             \
  EXPECT_EXIT(addAttr(LOOP, LoopAttrKind::NAME), ::testing::ExitedWithCode(1), \
              "error: cannot add attribute");

#define CHECK_GENERIC_FLAG(NAME, IRNAME)                                       \
  EXPECT_FALSE(hasAttr(*loop, LoopAttrKind::NAME));                            \
  addAttr(*loop, LoopAttrKind::NAME);                                          \
  EXPECT_TRUE(hasAttr(*loop, LoopAttrKind::NAME));                             \
  removeAttr(*loop, LoopAttrKind::NAME);                                       \
  EXPECT_FALSE(hasAttr(*loop, LoopAttrKind::NAME));

TEST(KitLoopAttrs, attrsGeneric) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  Loop *loop = *li.begin();

#define LOOP_ATTR_FLAG(NAME, IRNAME, TAPIRONLY) CHECK_GENERIC_FLAG(NAME, IRNAME)
#define LOOP_ATTR_ENUM(NAME, IRNAME, TAPIRONLY, TYPE)                          \
  CHECK_GENERIC(NAME, *loop, (TYPE)1)
#define LOOP_ATTR_I32(NAME, IRNAME, TAPIRONLY) CHECK_GENERIC(NAME, *loop, 31)
#define LOOP_ATTR_STR(NAME, IRNAME, TAPIRONLY)                                 \
  CHECK_GENERIC(NAME, *loop, "birkbeck")
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, flagAttrs) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  Loop *loop = *li.begin();

#define LOOP_ATTR_FLAG(NAME, IRNAME, TAPIRONLY)                                \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
                                                                               \
  add##NAME##Attr(*loop);                                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
                                                                               \
  add##NAME##Attr(*loop);                                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

#define CHECK_ACCESSORS(NAME, LOOP, VAL1, VAL2)                                \
  EXPECT_FALSE(has##NAME##Attr(LOOP));                                         \
                                                                               \
  add##NAME##Attr(LOOP, VAL1);                                                 \
  EXPECT_TRUE(has##NAME##Attr(LOOP));                                          \
  EXPECT_EQ(get##NAME##Attr(LOOP), (VAL1));                                    \
                                                                               \
  add##NAME##Attr(LOOP, VAL2);                                                 \
  EXPECT_TRUE(has##NAME##Attr(LOOP));                                          \
  EXPECT_EQ(get##NAME##Attr(LOOP), (VAL2));                                    \
                                                                               \
  remove##NAME##Attr(LOOP);                                                    \
  EXPECT_FALSE(has##NAME##Attr(LOOP));

TEST(KitLoopAttrs, enumAttrs) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  Loop *loop = *li.begin();

  // WARNING: This is somewhat risky because there is no guarantee that the
  // integer values 1 and 2 will be valid for every enum type that we may have
  // an attribute for. If this ever happens, change this test. It may be
  // sufficient to just check for some enum-valued attribute instead of all of
  // them.
#define LOOP_ATTR_ENUM(NAME, IRNAME, TAPIRONLY, TYPE)                          \
  CHECK_ACCESSORS(NAME, *loop, (TYPE)1, (TYPE)2)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, f32Test) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTR_F32(NAME, IRNAME, TAPIRONLY)                                 \
  CHECK_ACCESSORS(NAME, *loop, 3.14F, 2.71F)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, f64Test) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTR_F64(NAME, IRNAME, TAPIRONLY)                                 \
  CHECK_ACCESSORS(NAME, *loop, 3.1415926535, 2.71828)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, i32Test) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  Loop *loop = *li.begin();

#define LOOP_ATTR_I32(NAME, IRNAME, TAPIRONLY)                                 \
  CHECK_ACCESSORS(NAME, *loop, 83, 89)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, i64Test) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTR_I64(NAME, IRNAME, TAPIRONLY)                                 \
  CHECK_ACCESSORS(NAME, *loop, 71L, 73L)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, strTest) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  Loop *loop = *li.begin();

#define LOOP_ATTR_STR(NAME, IRNAME, TAPIRONLY)                                 \
  CHECK_ACCESSORS(NAME, *loop, "lse", "soas")
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

} // namespace
