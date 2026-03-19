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

template <typename T>
static void checkLoopGetMetadata(LLVMContext &ctx, LoopAttrKind attr, T val,
                                 unsigned llvmVal) {
  MDNode *md = getMDNodeForAttr(ctx, attr, val);
  auto *md0 = dyn_cast<MDString>(md->getOperand(0));
  auto *md1 = dyn_cast<ConstantAsMetadata>(md->getOperand(1));

  EXPECT_TRUE(md0);
  EXPECT_TRUE(md1);
  EXPECT_EQ(md->getNumOperands(), 2U);
  EXPECT_EQ(md0->getString(), getAttrName(attr));
  EXPECT_EQ(cast<ConstantInt>(md1->getValue())->getLimitedValue(), llvmVal);
}

static void checkLoopGetMetadata(LLVMContext &ctx, LoopAttrKind attr) {
  MDNode *md = getMDNodeForAttr(ctx, attr);
  EXPECT_EQ(md->getNumOperands(), 1U);

  auto *md0 = dyn_cast<MDString>(md->getOperand(0));
  EXPECT_TRUE(md0);
  EXPECT_EQ(md0->getString(), getAttrName(attr));
}

TEST(KitLoopAttrs, loopGetMetadata) {
  LLVMContext ctx;

  checkLoopGetMetadata(ctx, LoopAttrKind::LoweringEnabled);
  checkLoopGetMetadata(ctx, LoopAttrKind::Target, TTID::Serial, 1U);
  checkLoopGetMetadata(ctx, LoopAttrKind::PerfectDepth, 13, 13U);
}

TEST(KitLoopAttrs, loopAttrName) {
#define LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                                  \
  EXPECT_EQ(getAttrName(LoopAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(LoopAttrKind::NAME).starts_with("loop."));
#define TAPIR_LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                            \
  EXPECT_EQ(getAttrName(LoopAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(LoopAttrKind::NAME).starts_with("tapir.loop."));
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, loopAttrKind) {
  EXPECT_EQ(getLoopAttrKind("whoops"), std::nullopt);

#define LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                                  \
  EXPECT_EQ(getLoopAttrKind(IRNAME), LoopAttrKind::NAME);
#define TAPIR_LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                            \
  LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, loopAttrTapirOnly) {
#define LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                                  \
  EXPECT_FALSE(isAttrTapirOnly(LoopAttrKind::NAME));
#define TAPIR_LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                            \
  EXPECT_TRUE(isAttrTapirOnly(LoopAttrKind::NAME));
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

#define CHECK_GENERIC(NAME, LOOP, VAL)                                         \
  EXPECT_FALSE(hasAttr(LOOP, LoopAttrKind::NAME));                             \
  add##NAME##Attr(LOOP, VAL);                                                  \
  EXPECT_TRUE(hasAttr(LOOP, LoopAttrKind::NAME));                              \
  removeAttr(LOOP, LoopAttrKind::NAME);                                        \
  EXPECT_FALSE(hasAttr(LOOP, LoopAttrKind::NAME));

#define CHECK_ACCESSORS(NAME, INST, VAL1, VAL2)                                \
  EXPECT_FALSE(has##NAME##Attr(INST));                                         \
                                                                               \
  add##NAME##Attr(INST, VAL1);                                                 \
  EXPECT_TRUE(has##NAME##Attr(INST));                                          \
  EXPECT_EQ(get##NAME##Attr(INST), (VAL1));                                    \
                                                                               \
  add##NAME##Attr(INST, VAL2);                                                 \
  EXPECT_TRUE(has##NAME##Attr(INST));                                          \
  EXPECT_EQ(get##NAME##Attr(INST), (VAL2));                                    \
                                                                               \
  remove##NAME##Attr(INST);                                                    \
  EXPECT_FALSE(has##NAME##Attr(INST));

TEST(KitLoopAttrs, loopAttrsGeneric) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                      \
  EXPECT_FALSE(hasAttr(*loop, LoopAttrKind::NAME));                            \
  add##NAME##Attr(*loop);                                                      \
  EXPECT_TRUE(hasAttr(*loop, LoopAttrKind::NAME));                             \
  removeAttr(*loop, LoopAttrKind::NAME);                                       \
  EXPECT_FALSE(hasAttr(*loop, LoopAttrKind::NAME));
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
  LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)

#define LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)                                \
  CHECK_GENERIC(NAME, *loop, (TYPE)1)
#define LOOP_ATTRIBUTE_INT32(NAME, IRNAME) CHECK_GENERIC(NAME, *loop, 31)
#define LOOP_ATTRIBUTE_INT64(NAME, IRNAME) CHECK_GENERIC(NAME, *loop, 37L)
#define LOOP_ATTRIBUTE_STR(NAME, IRNAME) CHECK_GENERIC(NAME, *loop, "birkbeck")

#define TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)                          \
  LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)
#define TAPIR_LOOP_ATTRIBUTE_INT32(NAME, IRNAME)                               \
  LOOP_ATTRIBUTE_INT32(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_INT64(NAME, IRNAME)                               \
  LOOP_ATTRIBUTE_INT64(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME) LOOP_ATTRIBUTE_STR(NAME, IRNAME)

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, loopEnumAttrs) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

  // WARNING: This is somewhat risky because there is no guarantee that the
  // integer values 1 and 2 will be valid for every enum type that we may have
  // an attribute for. If this ever happens, change this test. It may be
  // sufficient to just check for some enum-valued attribute instead of all of
  // them.
#define LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)                                \
  CHECK_ACCESSORS(NAME, *loop, (TYPE)1, (TYPE)2)
#define TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)                          \
  LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, loopFlagAttrs) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                      \
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

#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
  LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, loopInt32Test) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTRIBUTE_INT32(NAME, IRNAME) CHECK_ACCESSORS(NAME, *loop, 83, 89)
#define TAPIR_LOOP_ATTRIBUTE_INT32(NAME, IRNAME)                               \
  LOOP_ATTRIBUTE_INT32(NAME, IRNAME)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, loopInt64Test) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTRIBUTE_INT64(NAME, IRNAME) CHECK_ACCESSORS(NAME, *loop, 3L, 5L)
#define TAPIR_LOOP_ATTRIBUTE_INT64(NAME, IRNAME)                               \
  LOOP_ATTRIBUTE_INT64(NAME, IRNAME)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, loopStrTest) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTRIBUTE_STR(NAME, IRNAME)                                       \
  CHECK_ACCESSORS(NAME, *loop, "lse", "soas")
#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME) LOOP_ATTRIBUTE_STR(NAME, IRNAME)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

} // namespace
