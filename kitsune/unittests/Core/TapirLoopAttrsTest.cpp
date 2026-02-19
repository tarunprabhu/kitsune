//===- TapirLoopAttrsTest.cpp - Unit tests for tapir loop attributes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TapirLoopAttrs.h"
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

static bool hasAttr(const MDNode &loopMD, StringRef attrName) {
  for (unsigned i = 1, ie = loopMD.getNumOperands(); i < ie; ++i)
    if (auto *md = dyn_cast<MDNode>(loopMD.getOperand(i)))
      if (auto *mdStr = dyn_cast<MDString>(md->getOperand(0)))
        if (mdStr->getString() == attrName)
          return true;
  return false;
}

static bool hasLoopAttr(const Loop &loop, StringRef attrName) {
  return hasAttr(*loop.getLoopID(), attrName);
}

TEST(TapirLoopAttrsTest, tapirLoopGetMetadata) {
  LLVMContext ctx;

  MDNode *md0 =
      getMetadataForTapirLoopAttr(ctx, TapirLoopAttrKind::Target, TTID::Serial);
  EXPECT_EQ(md0->getNumOperands(), 2U);
  EXPECT_EQ(cast<MDString>(md0->getOperand(0))->getString(),
            "tapir.loop.target");
  Constant *c0 = cast<ConstantAsMetadata>(md0->getOperand(1))->getValue();
  EXPECT_EQ(cast<ConstantInt>(c0)->getLimitedValue(), 1U);

  MDNode *md1 =
      getMetadataForTapirLoopAttr(ctx, TapirLoopAttrKind::LoweringEnabled);
  EXPECT_EQ(md1->getNumOperands(), 2U);
  EXPECT_EQ(cast<MDString>(md1->getOperand(0))->getString(),
            "tapir.loop.lowering.enabled");
  Constant *c1 = cast<ConstantAsMetadata>(md1->getOperand(1))->getValue();
  EXPECT_EQ(cast<ConstantInt>(c1)->getLimitedValue(), 1U);

  MDNode *md2 =
      getMetadataForTapirLoopAttr(ctx, TapirLoopAttrKind::PerfectDepth, 13U);
  EXPECT_EQ(md2->getNumOperands(), 2U);
  EXPECT_EQ(cast<MDString>(md2->getOperand(0))->getString(),
            "tapir.loop.perfect.depth");
  Constant *c2 = cast<ConstantAsMetadata>(md2->getOperand(1))->getValue();
  EXPECT_EQ(cast<ConstantInt>(c2)->getLimitedValue(), 13U);
}

TEST(TapirLoopAttrsTest, tapirLoopAttrName) {
#define TAPIR_LOOP_ATTR(NAME, IRNAME)                                          \
  EXPECT_EQ(getTapirLoopAttrName(TapirLoopAttrKind::NAME), IRNAME);

#define TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)                          \
  TAPIR_LOOP_ATTR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_UINT(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_ULONG(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME) TAPIR_LOOP_ATTR(NAME, IRNAME)

#include "kitsune/Core/TapirLoopAttrs.inc"

#undef TAPIR_LOOP_ATTRIBUTE_ULONG
#undef TAPIR_LOOP_ATTRIBUTE_UINT
#undef TAPIR_LOOP_ATTRIBUTE_STR
#undef TAPIR_LOOP_ATTRIBUTE_FLAG
#undef TAPIR_LOOP_ATTRIBUTE_ENUM
}

TEST(TapirLoopAttrsTest, tapirLoopFlagAttrs) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = li.getLoopsInPreorder().front();

#define TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)
#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_UINT(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_ULONG(NAME, IRNAME)
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

#include "kitsune/Core/TapirLoopAttrs.inc"

#undef TAPIR_LOOP_ATTRIBUTE_ULONG
#undef TAPIR_LOOP_ATTRIBUTE_UINT
#undef TAPIR_LOOP_ATTRIBUTE_STR
#undef TAPIR_LOOP_ATTRIBUTE_FLAG
#undef TAPIR_LOOP_ATTRIBUTE_ENUM
}

TEST(TapirLoopAttrsTest, tapirLoopStrTest) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = li.getLoopsInPreorder().front();

#define TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_UINT(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_ULONG(NAME, IRNAME)
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

#include "kitsune/Core/TapirLoopAttrs.inc"

#undef TAPIR_LOOP_ATTRIBUTE_ULONG
#undef TAPIR_LOOP_ATTRIBUTE_UINT
#undef TAPIR_LOOP_ATTRIBUTE_STR
#undef TAPIR_LOOP_ATTRIBUTE_FLAG
#undef TAPIR_LOOP_ATTRIBUTE_ENUM
}

TEST(TapirLoopAttrsTest, tapirLoopUintTest) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = li.getLoopsInPreorder().front();

#define TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_ULONG(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_UINT(NAME, IRNAME)                                \
  EXPECT_FALSE(hasTapirLoop##NAME##Attr(*loop));                               \
  EXPECT_EQ(getTapirLoop##NAME##Attr(*loop), std::nullopt);                    \
                                                                               \
  addTapirLoop##NAME##Attr(*loop, 42U);                                        \
  EXPECT_TRUE(hasTapirLoop##NAME##Attr(*loop));                                \
  EXPECT_EQ(*getTapirLoop##NAME##Attr(*loop), 42U);                            \
                                                                               \
  addTapirLoop##NAME##Attr(*loop, 97U);                                        \
  EXPECT_TRUE(hasTapirLoop##NAME##Attr(*loop));                                \
  EXPECT_EQ(*getTapirLoop##NAME##Attr(*loop), 97U);                            \
                                                                               \
  removeTapirLoop##NAME##Attr(*loop);                                          \
  EXPECT_FALSE(hasTapirLoop##NAME##Attr(*loop));                               \
  EXPECT_EQ(getTapirLoop##NAME##Attr(*loop), std::nullopt);

#include "kitsune/Core/TapirLoopAttrs.inc"

#undef TAPIR_LOOP_ATTRIBUTE_ULONG
#undef TAPIR_LOOP_ATTRIBUTE_UINT
#undef TAPIR_LOOP_ATTRIBUTE_STR
#undef TAPIR_LOOP_ATTRIBUTE_FLAG
#undef TAPIR_LOOP_ATTRIBUTE_ENUM
}

TEST(TapirLoopAttrsTest, tapirLoopULongTest) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = li.getLoopsInPreorder().front();

#define TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)
#define TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_UINT(NAME, IRNAME)
#define TAPIR_LOOP_ATTRIBUTE_ULONG(NAME, IRNAME)                               \
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
  removeTapirLoop##NAME##Attr(*loop);                                          \
  EXPECT_FALSE(hasTapirLoop##NAME##Attr(*loop));                               \
  EXPECT_EQ(getTapirLoop##NAME##Attr(*loop), std::nullopt);

#include "kitsune/Core/TapirLoopAttrs.inc"

#undef TAPIR_LOOP_ATTRIBUTE_ULONG
#undef TAPIR_LOOP_ATTRIBUTE_UINT
#undef TAPIR_LOOP_ATTRIBUTE_STR
#undef TAPIR_LOOP_ATTRIBUTE_FLAG
#undef TAPIR_LOOP_ATTRIBUTE_ENUM
}

TEST(LoopUtilsTest, clearTapirLoopAttrs) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, loop1);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);

  Loop *loop = li.getLoopsInPreorder().front();
  MDNode *md = MDNode::get(ctx, {MDString::get(ctx, "loop.unroll")});
  addTapirLoopLoweringEnabledAttr(*loop);
  addTapirLoopPerfectDepthAttr(*loop, 17U);
  addTapirLoopTargetAttr(*loop, TTID::Serial);
  loop->setLoopID(
      makePostTransformationMetadata(ctx, loop->getLoopID(), {}, {md}));

  EXPECT_TRUE(hasTapirLoopLoweringEnabledAttr(*loop));
  EXPECT_TRUE(hasTapirLoopPerfectDepthAttr(*loop));
  EXPECT_TRUE(hasTapirLoopTargetAttr(*loop));
  EXPECT_TRUE(hasLoopAttr(*loop, "loop.unroll"));
  EXPECT_EQ(loop->getLoopID()->getNumOperands(), 5U);

  clearTapirLoopAttrs(*loop);
  EXPECT_FALSE(hasTapirLoopLoweringEnabledAttr(*loop));
  EXPECT_FALSE(hasTapirLoopPerfectDepthAttr(*loop));
  EXPECT_FALSE(hasTapirLoopTargetAttr(*loop));
  EXPECT_TRUE(hasLoopAttr(*loop, "loop.unroll"));
  EXPECT_EQ(loop->getLoopID()->getNumOperands(), 2U);
}
