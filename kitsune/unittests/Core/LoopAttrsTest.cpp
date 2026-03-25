//===- LoopAttrsTest.cpp - Unit tests for Kitsune's loop attributes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/LoopAttrs.h"
#include "TestAttrsCommon.h"
#include "TestValues.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

// The standard accessors do not allow us to create invalid attributes. To
// create one, we have to know how these are added to the function. This is not
// unreasonable since the create functions are a fairly thin wrappers around
// LLVM's existing support.
static void addMetadata(Loop &loop, StringRef name, ArrayRef<Metadata *> ops) {
  LLVMContext &ctx = getContext(loop);
  Metadata *mdTag = MDString::get(ctx, name);

  SmallVector<Metadata *, 8> mdOps = {mdTag};
  mdOps.append(ops.begin(), ops.end());

  MDNode *md = MDNode::get(ctx, mdOps);
  MDNode *loopID = loop.getLoopID();
  MDNode *newLoopID = makePostTransformationMetadata(ctx, loopID, {name}, {md});

  loop.setLoopID(newLoopID);
}

// Create metadata consisting of `n` "empty" operands.
static void addMetadata(Loop &loop, StringRef name, unsigned n) {
  LLVMContext &ctx = getContext(loop);
  MDNode *mdEmpty = MDNode::get(ctx, {});
  SmallVector<Metadata *, 8> ops;

  ops.append(n, mdEmpty);
  addMetadata(loop, name, ops);
}

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
  LLVMContext ctx;
#define CHECK_METADATA_1(ATTR, VAL, EXP)                                       \
  {                                                                            \
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
}

TEST(KitLoopAttrs, attrName) {
#define LOOP_ATTR(NAME, IRNAME, TYPE)                                          \
  EXPECT_EQ(getAttrName(LoopAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(LoopAttrKind::NAME).starts_with("tapir.loop."));
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attrKind) {
  EXPECT_EQ(getLoopAttrKind("wolfson"), std::nullopt);
#define LOOP_ATTR(NAME, IRNAME, TYPE)                                          \
  EXPECT_EQ(getLoopAttrKind(IRNAME), LoopAttrKind::NAME);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

#define DECLS(OS, OBJ)                                                         \
  std::string buf;                                                             \
  raw_string_ostream os(buf);                                                  \
  LLVMContext ctx;                                                             \
  std::unique_ptr<Module> m = parseIR(ctx, ll);                                \
  Function *f = m->getFunction("f");                                           \
  DominatorTree dt(*f);                                                        \
  LoopInfo li(dt);                                                             \
  [[maybe_unused]] Loop OBJ = *li.begin()

TEST(KitLoopAttrs, verifyGeneric) {
  DECLS(os, *loop);
#define LOOP_ATTR_0(NAME, IRNAME)                                              \
  TEST_GENERIC_VERIFY_0(os, *loop, LoopAttrKind, NAME, IRNAME);
#define LOOP_ATTR(NAME, IRNAME, TYPE)                                          \
  TEST_GENERIC_VERIFY_N(os, *loop, LoopAttrKind, NAME, IRNAME);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attrsGeneric) {
  DECLS(os, *loop);

#define LOOP_ATTR_0(NAME, IRNAME)                                              \
  TEST_GENERIC_ATTR_0(*loop, LoopAttrKind, NAME);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_0(NAME, IRNAME)
#define LOOP_ATTR(NAME, IRNAME, TYPE)                                          \
  TEST_GENERIC_ATTR_N(*loop, LoopAttrKind, NAME);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr0) {
  DECLS(os, *loop);
#define LOOP_ATTR_0(NAME, IRNAME) TEST_ATTR_0(os, *loop, NAME, IRNAME);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr1) {
  DECLS(os, *loop);
#define LOOP_ATTR_1(NAME, IRNAME, TYPE)                                        \
  TEST_ATTR_1(os, *loop, NAME, IRNAME, TYPE);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr2) {
  DECLS(os, *loop);
#define LOOP_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  TEST_ATTR_2(os, *loop, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr3) {
  DECLS(os, *loop);
#define LOOP_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  TEST_ATTR_3(os, *loop, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,   \
              ETY2, ENAME2, EN2);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr4) {
  DECLS(os, *loop);
#define LOOP_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  TEST_ATTR_4(os, *loop, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,   \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr5) {
  DECLS(os, *loop);
#define LOOP_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  TEST_ATTR_5(os, *loop, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,   \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr6) {
  DECLS(os, *loop);
#define LOOP_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  TEST_ATTR_6(os, *loop, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,   \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
              ENAME5, EN5);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr7) {
  DECLS(os, *loop);
#define LOOP_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  TEST_ATTR_7(os, *loop, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,   \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
              ENAME5, EN5, ETY6, ENAME6, EN6);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr8) {
  DECLS(os, *loop);
#define LOOP_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  TEST_ATTR_8(os, *loop, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,   \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
              ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

} // namespace
