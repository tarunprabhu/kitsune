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

constexpr StringRef loop1 = R"(
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
  MDNode *md = getMetadataForAttr(ctx, attr, val);
  auto *md0 = dyn_cast<MDString>(md->getOperand(0));
  auto *md1 = dyn_cast<ConstantAsMetadata>(md->getOperand(1));

  EXPECT_TRUE(md0);
  EXPECT_TRUE(md1);
  EXPECT_EQ(md->getNumOperands(), 2U);
  EXPECT_EQ(md0->getString(), getAttrName(attr));
  EXPECT_EQ(cast<ConstantInt>(md1->getValue())->getLimitedValue(), llvmVal);
}

static void checkLoopGetMetadata(LLVMContext &ctx, LoopAttrKind attr) {
  MDNode *md = getMetadataForAttr(ctx, attr);
  auto *md0 = dyn_cast<MDString>(md->getOperand(0));
  auto *md1 = dyn_cast<ConstantAsMetadata>(md->getOperand(1));

  EXPECT_TRUE(md0);
  EXPECT_TRUE(md1);
  EXPECT_EQ(md->getNumOperands(), 2U);
  EXPECT_EQ(md0->getString(), getAttrName(attr));
  EXPECT_EQ(cast<ConstantInt>(md1->getValue())->getLimitedValue(), 1U);
}

TEST(KitLoopAttrs, loopGetMetadata) {
  LLVMContext ctx;

  checkLoopGetMetadata(ctx, LoopAttrKind::LoweringEnabled);
  checkLoopGetMetadata(ctx, LoopAttrKind::Target, TTID::Serial, 1U);
  checkLoopGetMetadata(ctx, LoopAttrKind::PerfectDepth, 13, 13U);
}

TEST(KitLoopAttrs, loopAttrName) {
#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                      \
  EXPECT_EQ(getAttrName(LoopAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(LoopAttrKind::NAME).starts_with("loop."));
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
  EXPECT_EQ(getAttrName(LoopAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(LoopAttrKind::NAME).starts_with("tapir.loop."));
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, loopAttrKind) {
  EXPECT_EQ(getLoopAttrKind("whoops"), std::nullopt);

#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                      \
  EXPECT_EQ(getLoopAttrKind(IRNAME), LoopAttrKind::NAME);
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
  LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)
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

TEST(KitLoopAttrs, loopAttrsGeneric) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = li.getLoopsInPreorder().front();

  auto checkCommon = [](Loop &loop, LoopAttrKind attr) -> void {
    EXPECT_TRUE(hasAttr(loop, attr));
    removeAttr(loop, attr);
    EXPECT_FALSE(hasAttr(loop, attr));
  };

#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                      \
  EXPECT_FALSE(hasAttr(*loop, LoopAttrKind::NAME));                            \
  add##NAME##Attr(*loop);                                                      \
  checkCommon(*loop, LoopAttrKind::NAME);

#define LOOP_ATTRIBUTE_INT32(NAME, IRNAME)                                     \
  EXPECT_FALSE(hasAttr(*loop, LoopAttrKind::NAME));                            \
  add##NAME##Attr(*loop, 67);                                                  \
  checkCommon(*loop, LoopAttrKind::NAME);

#define LOOP_ATTRIBUTE_INT64(NAME, IRNAME)                                     \
  EXPECT_FALSE(hasAttr(*loop, LoopAttrKind::NAME));                            \
  add##NAME##Attr(*loop, 67L);                                                 \
  checkCommon(*loop, LoopAttrKind::NAME);

#define LOOP_ATTRIBUTE_STR(NAME, IRNAME)                                       \
  EXPECT_FALSE(hasAttr(*loop, LoopAttrKind::NAME));                            \
  add##NAME##Attr(*loop, "67");                                                \
  checkCommon(*loop, LoopAttrKind::NAME);

#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
  EXPECT_FALSE(hasAttr(*loop, LoopAttrKind::NAME));                            \
  add##NAME##Attr(*loop);                                                      \
  checkCommon(*loop, LoopAttrKind::NAME);

#define TAPIR_LOOP_ATTRIBUTE_INT32(NAME, IRNAME)                               \
  EXPECT_FALSE(hasAttr(*loop, LoopAttrKind::NAME));                            \
  add##NAME##Attr(*loop, 67);                                                  \
  checkCommon(*loop, LoopAttrKind::NAME);

#define TAPIR_LOOP_ATTRIBUTE_INT64(NAME, IRNAME)                               \
  EXPECT_FALSE(hasAttr(*loop, LoopAttrKind::NAME));                            \
  add##NAME##Attr(*loop, 67L);                                                 \
  checkCommon(*loop, LoopAttrKind::NAME);

#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME)                                 \
  EXPECT_FALSE(hasAttr(*loop, LoopAttrKind::NAME));                            \
  addTapriLoop##NAME##Attr(*loop, "67");                                       \
  checkCommon(*loop, LoopAttrKind::NAME);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, loopFlagAttrs) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = li.getLoopsInPreorder().front();

#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                      \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                    \
                                                                               \
  add##NAME##Attr(*loop);                                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                     \
                                                                               \
  add##NAME##Attr(*loop);                                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                     \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));

#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
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

TEST(KitLoopAttrs, loopInt32Test) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = li.getLoopsInPreorder().front();

#define LOOP_ATTRIBUTE_INT32(NAME, IRNAME)                                     \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_EQ(get##NAME##Attr(*loop), std::nullopt);                             \
                                                                               \
  add##NAME##Attr(*loop, 42);                                                  \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(*get##NAME##Attr(*loop), 42);                                      \
                                                                               \
  add##NAME##Attr(*loop, 97);                                                  \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(*get##NAME##Attr(*loop), 97);                                      \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_EQ(get##NAME##Attr(*loop), std::nullopt);

#define TAPIR_LOOP_ATTRIBUTE_INT32(NAME, IRNAME)                               \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_EQ(get##NAME##Attr(*loop), std::nullopt);                             \
                                                                               \
  add##NAME##Attr(*loop, 42);                                                  \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(*get##NAME##Attr(*loop), 42);                                      \
                                                                               \
  add##NAME##Attr(*loop, 97);                                                  \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(*get##NAME##Attr(*loop), 97);                                      \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_EQ(get##NAME##Attr(*loop), std::nullopt);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, loopInt64Test) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = li.getLoopsInPreorder().front();

#define LOOP_ATTRIBUTE_INT64(NAME, IRNAME)                                     \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_EQ(get##NAME##Attr(*loop), std::nullopt);                             \
                                                                               \
  add##NAME##Attr(*loop, 42L);                                                 \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(*get##NAME##Attr(*loop), 42L);                                     \
                                                                               \
  add##NAME##Attr(*loop, 97L);                                                 \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(*get##NAME##Attr(*loop), 97L);                                     \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_EQ(get##NAME##Attr(*loop), std::nullopt);

#define TAPIR_LOOP_ATTRIBUTE_INT64(NAME, IRNAME)                               \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_EQ(get##NAME##Attr(*loop), std::nullopt);                             \
                                                                               \
  add##NAME##Attr(*loop, 42L);                                                 \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(*get##NAME##Attr(*loop), 42L);                                     \
                                                                               \
  add##NAME##Attr(*loop, 97L);                                                 \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(*get##NAME##Attr(*loop), 97L);                                     \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_EQ(get##NAME##Attr(*loop), std::nullopt);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, loopStrTest) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = li.getLoopsInPreorder().front();

#define LOOP_ATTRIBUTE_STR(NAME, IRNAME)                                       \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_EQ(get##NAME##Attr(*loop), std::nullopt);                             \
                                                                               \
  add##NAME##Attr(*loop, "42");                                                \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(*get##NAME##Attr(*loop), "42");                                    \
                                                                               \
  add##NAME##Attr(*loop, "97");                                                \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(*get##NAME##Attr(*loop), "97");                                    \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_EQ(get##NAME##Attr(*loop), std::nullopt);

#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME)                                 \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_EQ(get##NAME##Attr(*loop), std::nullopt);                             \
                                                                               \
  add##NAME##Attr(*loop, "42");                                                \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(*get##NAME##Attr(*loop), "42");                                    \
                                                                               \
  add##NAME##Attr(*loop, "97");                                                \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(*get##NAME##Attr(*loop), "97");                                    \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_EQ(get##NAME##Attr(*loop), std::nullopt);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

} // namespace
