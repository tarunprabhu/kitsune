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

TEST(LoopAttrsTest, loopGetMetadata) {
  LLVMContext ctx;

  MDNode *md0 = getMetadataForLoopAttr(ctx, LoopAttrKind::Target, TTID::Serial);
  EXPECT_EQ(md0->getNumOperands(), 2U);
  EXPECT_EQ(cast<MDString>(md0->getOperand(0))->getString(),
            getLoopAttrName(LoopAttrKind::Target));
  Constant *c0 = cast<ConstantAsMetadata>(md0->getOperand(1))->getValue();
  EXPECT_EQ(cast<ConstantInt>(c0)->getLimitedValue(), 1U);

  MDNode *md1 = getMetadataForLoopAttr(ctx, LoopAttrKind::LoweringEnabled);
  EXPECT_EQ(md1->getNumOperands(), 2U);
  EXPECT_EQ(cast<MDString>(md1->getOperand(0))->getString(),
            getLoopAttrName(LoopAttrKind::LoweringEnabled));
  Constant *c1 = cast<ConstantAsMetadata>(md1->getOperand(1))->getValue();
  EXPECT_EQ(cast<ConstantInt>(c1)->getLimitedValue(), 1U);

  MDNode *md2 = getMetadataForLoopAttr(ctx, LoopAttrKind::PerfectDepth, 13);
  EXPECT_EQ(md2->getNumOperands(), 2U);
  EXPECT_EQ(cast<MDString>(md2->getOperand(0))->getString(),
            getLoopAttrName(LoopAttrKind::PerfectDepth));
  Constant *c2 = cast<ConstantAsMetadata>(md2->getOperand(1))->getValue();
  EXPECT_EQ(cast<ConstantInt>(c2)->getLimitedValue(), 13U);
}

TEST(LoopAttrsTest, loopAttrName) {
#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                      \
  EXPECT_EQ(getLoopAttrName(LoopAttrKind::NAME), IRNAME);                      \
  EXPECT_TRUE(getLoopAttrName(LoopAttrKind::NAME).starts_with("loop."));
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
  EXPECT_EQ(getLoopAttrName(LoopAttrKind::NAME), IRNAME);                   \
  EXPECT_TRUE(getLoopAttrName(LoopAttrKind::NAME).starts_with("tapir.loop."));
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(LoopAttrsTest, loopAttrKind) {
  EXPECT_EQ(getLoopAttrKind("whoops"), std::nullopt);

#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                      \
  EXPECT_EQ(getLoopAttrKind(IRNAME), LoopAttrKind::NAME);
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
  LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(LoopAttrsTest, loopAttrTapirOnly) {
#define LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                                  \
  EXPECT_FALSE(isLoopAttrTapirOnly(LoopAttrKind::NAME));
#define TAPIR_LOOP_ATTR(NAME, TYPE, IRNAME, IRTYPE)                            \
  EXPECT_TRUE(isLoopAttrTapirOnly(LoopAttrKind::NAME));
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(LoopAttrsTest, loopFlagAttrs) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = li.getLoopsInPreorder().front();

#define LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                      \
  EXPECT_FALSE(hasLoop##NAME##Attr(*loop));                                    \
                                                                               \
  addLoop##NAME##Attr(*loop);                                                  \
  EXPECT_TRUE(hasLoop##NAME##Attr(*loop));                                     \
                                                                               \
  addLoop##NAME##Attr(*loop);                                                  \
  EXPECT_TRUE(hasLoop##NAME##Attr(*loop));                                     \
                                                                               \
  removeLoop##NAME##Attr(*loop);                                               \
  EXPECT_FALSE(hasLoop##NAME##Attr(*loop));

#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)                                \
  EXPECT_FALSE(hasTapirLoop##NAME##Attr(*loop));                               \
                                                                               \
  addTapirLoop##NAME##Attr(*loop);                                             \
  EXPECT_TRUE(hasTapirLoop##NAME##Attr(*loop));                                \
                                                                               \
  addTapirLoop##NAME##Attr(*loop);                                             \
  EXPECT_TRUE(hasTapirLoop##NAME##Attr(*loop));                                \
                                                                               \
  removeTapirLoop##NAME##Attr(*loop);                                          \
  EXPECT_FALSE(hasTapirLoop##NAME##Attr(*loop));

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(LoopAttrsTest, loopInt32Test) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = li.getLoopsInPreorder().front();

#define LOOP_ATTRIBUTE_INT32(NAME, IRNAME)                                     \
  EXPECT_FALSE(hasLoop##NAME##Attr(*loop));                                    \
  EXPECT_EQ(getLoop##NAME##Attr(*loop), std::nullopt);                         \
                                                                               \
  addLoop##NAME##Attr(*loop, 42);                                              \
  EXPECT_TRUE(hasLoop##NAME##Attr(*loop));                                     \
  EXPECT_EQ(*getLoop##NAME##Attr(*loop), 42);                                  \
                                                                               \
  addLoop##NAME##Attr(*loop, 97);                                              \
  EXPECT_TRUE(hasLoop##NAME##Attr(*loop));                                     \
  EXPECT_EQ(*getLoop##NAME##Attr(*loop), 97);                                  \
                                                                               \
  removeLoop##NAME##Attr(*loop);                                               \
  EXPECT_FALSE(hasLoop##NAME##Attr(*loop));                                    \
  EXPECT_EQ(getLoop##NAME##Attr(*loop), std::nullopt);

#define TAPIR_LOOP_ATTRIBUTE_INT32(NAME, IRNAME)                               \
  EXPECT_FALSE(hasTapirLoop##NAME##Attr(*loop));                               \
  EXPECT_EQ(getTapirLoop##NAME##Attr(*loop), std::nullopt);                    \
                                                                               \
  addTapirLoop##NAME##Attr(*loop, 42);                                         \
  EXPECT_TRUE(hasTapirLoop##NAME##Attr(*loop));                                \
  EXPECT_EQ(*getTapirLoop##NAME##Attr(*loop), 42);                             \
                                                                               \
  addTapirLoop##NAME##Attr(*loop, 97);                                         \
  EXPECT_TRUE(hasTapirLoop##NAME##Attr(*loop));                                \
  EXPECT_EQ(*getTapirLoop##NAME##Attr(*loop), 97);                             \
                                                                               \
  removeTapirLoop##NAME##Attr(*loop);                                          \
  EXPECT_FALSE(hasTapirLoop##NAME##Attr(*loop));                               \
  EXPECT_EQ(getTapirLoop##NAME##Attr(*loop), std::nullopt);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(LoopAttrsTest, loopInt64Test) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = li.getLoopsInPreorder().front();

#define LOOP_ATTRIBUTE_INT64(NAME, IRNAME)                                     \
  EXPECT_FALSE(hasLoop##NAME##Attr(*loop));                                    \
  EXPECT_EQ(getLoop##NAME##Attr(*loop), std::nullopt);                         \
                                                                               \
  addLoop##NAME##Attr(*loop, 42L);                                             \
  EXPECT_TRUE(hasLoop##NAME##Attr(*loop));                                     \
  EXPECT_EQ(*getLoop##NAME##Attr(*loop), 42L);                                 \
                                                                               \
  addLoop##NAME##Attr(*loop, 97L);                                             \
  EXPECT_TRUE(hasLoop##NAME##Attr(*loop));                                     \
  EXPECT_EQ(*getLoop##NAME##Attr(*loop), 97L);                                 \
                                                                               \
  removeLoop##NAME##Attr(*loop);                                               \
  EXPECT_FALSE(hasLoop##NAME##Attr(*loop));                                    \
  EXPECT_EQ(getLoop##NAME##Attr(*loop), std::nullopt);

#define TAPIR_LOOP_ATTRIBUTE_INT64(NAME, IRNAME)                               \
  EXPECT_FALSE(hasTapirLoop##NAME##Attr(*loop));                               \
  EXPECT_EQ(getTapirLoop##NAME##Attr(*loop), std::nullopt);                    \
                                                                               \
  addTapirLoop##NAME##Attr(*loop, 42L);                                        \
  EXPECT_TRUE(hasTapirLoop##NAME##Attr(*loop));                                \
  EXPECT_EQ(*getTapirLoop##NAME##Attr(*loop), 42L);                            \
                                                                               \
  addTapirLoop##NAME##Attr(*loop, 97L);                                        \
  EXPECT_TRUE(hasTapirLoop##NAME##Attr(*loop));                                \
  EXPECT_EQ(*getTapirLoop##NAME##Attr(*loop), 97L);                            \
                                                                               \
  removeLoop##NAME##Attr(*loop);                                               \
  EXPECT_FALSE(hasTapirLoop##NAME##Attr(*loop));                               \
  EXPECT_EQ(getTapirLoop##NAME##Attr(*loop), std::nullopt);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(LoopAttrsTest, loopStrTest) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = li.getLoopsInPreorder().front();

#define LOOP_ATTRIBUTE_STR(NAME, IRNAME)                                       \
  EXPECT_FALSE(hasLoop##NAME##Attr(*loop));                                    \
  EXPECT_EQ(getLoop##NAME##Attr(*loop), std::nullopt);                         \
                                                                               \
  addLoop##NAME##Attr(*loop, "42");                                            \
  EXPECT_TRUE(hasLoop##NAME##Attr(*loop));                                     \
  EXPECT_EQ(*getLoop##NAME##Attr(*loop), "42");                                \
                                                                               \
  addLoop##NAME##Attr(*loop, "97");                                            \
  EXPECT_TRUE(hasLoop##NAME##Attr(*loop));                                     \
  EXPECT_EQ(*getLoop##NAME##Attr(*loop), "97");                                \
                                                                               \
  removeLoop##NAME##Attr(*loop);                                               \
  EXPECT_FALSE(hasLoop##NAME##Attr(*loop));                                    \
  EXPECT_EQ(getLoop##NAME##Attr(*loop), std::nullopt);

#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME)                                 \
  EXPECT_FALSE(hasTapirLoop##NAME##Attr(*loop));                               \
  EXPECT_EQ(getTapirLoop##NAME##Attr(*loop), std::nullopt);                    \
                                                                               \
  addTapirLoop##NAME##Attr(*loop, "42");                                       \
  EXPECT_TRUE(hasTapirLoop##NAME##Attr(*loop));                                \
  EXPECT_EQ(*getTapirLoop##NAME##Attr(*loop), "42");                           \
                                                                               \
  addTapirLoop##NAME##Attr(*loop, "97");                                       \
  EXPECT_TRUE(hasTapirLoop##NAME##Attr(*loop));                                \
  EXPECT_EQ(*getTapirLoop##NAME##Attr(*loop), "97");                           \
                                                                               \
  removeTapirLoop##NAME##Attr(*loop);                                          \
  EXPECT_FALSE(hasTapirLoop##NAME##Attr(*loop));                               \
  EXPECT_EQ(getTapirLoop##NAME##Attr(*loop), std::nullopt);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}
