//===- LoopAttrsTest.cpp - Unit tests for Kitsune's loop attributes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/LoopAttrs.h"
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

TEST(KitLoopAttrs, verify) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTR_0(NAME, IRNAME)                                              \
  addMetadata(*loop, IRNAME, MDString::get(ctx, ""));                          \
                                                                               \
  EXPECT_FALSE(verifyAttr(*loop, LoopAttrKind::NAME));                         \
  EXPECT_FALSE(verify##NAME##Attr(*loop));                                     \
                                                                               \
  remove##NAME##Attr(*loop);

#define LOOP_ATTR(NAME, IRNAME, TYPE)                                          \
  addMetadata(*loop, IRNAME, {});                                              \
                                                                               \
  EXPECT_FALSE(verifyAttr(*loop, LoopAttrKind::NAME));                         \
  EXPECT_FALSE(verify##NAME##Attr(*loop));                                     \
                                                                               \
  remove##NAME##Attr(*loop);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attrsGeneric) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTR_0(NAME, IRNAME)                                              \
  EXPECT_FALSE(hasAttr(*loop, LoopAttrKind::NAME));                            \
  addAttr(*loop, LoopAttrKind::NAME);                                          \
  EXPECT_TRUE(hasAttr(*loop, LoopAttrKind::NAME));                             \
  addAttr(*loop, LoopAttrKind::NAME);                                          \
  EXPECT_TRUE(hasAttr(*loop, LoopAttrKind::NAME));                             \
  removeAttr(*loop, LoopAttrKind::NAME);                                       \
  EXPECT_FALSE(hasAttr(*loop, LoopAttrKind::NAME));
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_0(NAME, IRNAME)
#define LOOP_ATTR(NAME, IRNAME, TYPE)                                          \
  EXPECT_EXIT(addAttr(*loop, LoopAttrKind::NAME),                              \
              ::testing::ExitedWithCode(1), "error: cannot add attribute");
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr0) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTR_0(NAME, IRNAME)                                              \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
                                                                               \
  add##NAME##Attr(*loop);                                                      \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
                                                                               \
  add##NAME##Attr(*loop);                                                      \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
                                                                               \
  addMetadata(*loop, IRNAME, 0);                                               \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  remove##NAME##Attr(*loop);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr1) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTR_1(NAME, IRNAME, TYPE)                                        \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
                                                                               \
  add##NAME##Attr(*loop, get<TYPE>(0));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##NAME##Attr(*loop), get<TYPE>(0));                             \
                                                                               \
  add##NAME##Attr(*loop, get<TYPE>(1));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##NAME##Attr(*loop), get<TYPE>(1));                             \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
                                                                               \
  addMetadata(*loop, IRNAME, 1);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*loop));                                     \
  remove##NAME##Attr(*loop);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr2) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
                                                                               \
  add##NAME##Attr(*loop, get<ETY0>(0), get<ETY1>(1));                          \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*loop), get<ETY0>(0));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*loop), get<ETY1>(1));               \
                                                                               \
  add##NAME##Attr(*loop, get<ETY0>(1), get<ETY1>(0));                          \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*loop), get<ETY0>(1));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*loop), get<ETY1>(0));               \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
                                                                               \
  addMetadata(*loop, IRNAME, 2);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*loop));                                     \
  remove##NAME##Attr(*loop);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr3) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
                                                                               \
  add##NAME##Attr(*loop, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2));            \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*loop), get<ETY0>(0));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*loop), get<ETY1>(1));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*loop), get<ETY2>(2));               \
                                                                               \
  add##NAME##Attr(*loop, get<ETY0>(2), get<ETY1>(1), get<ETY2>(0));            \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*loop), get<ETY0>(2));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*loop), get<ETY1>(1));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*loop), get<ETY2>(0));               \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
                                                                               \
  addMetadata(*loop, IRNAME, 3);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*loop));                                     \
  remove##NAME##Attr(*loop);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr4) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
                                                                               \
  add##NAME##Attr(*loop, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                  get<ETY3>(3));                                               \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*loop), get<ETY0>(0));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*loop), get<ETY1>(1));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*loop), get<ETY2>(2));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*loop), get<ETY3>(3));               \
                                                                               \
  add##NAME##Attr(*loop, get<ETY0>(3), get<ETY1>(2), get<ETY2>(1),             \
                  get<ETY3>(0));                                               \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*loop), get<ETY0>(3));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*loop), get<ETY1>(2));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*loop), get<ETY2>(1));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*loop), get<ETY3>(0));               \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
                                                                               \
  addMetadata(*loop, IRNAME, 4);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*loop));                                     \
  remove##NAME##Attr(*loop);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr5) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
                                                                               \
  add##NAME##Attr(*loop, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                  get<ETY3>(3), get<ETY4>(4));                                 \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*loop), get<ETY0>(0));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*loop), get<ETY1>(1));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*loop), get<ETY2>(2));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*loop), get<ETY3>(3));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*loop), get<ETY4>(4));               \
                                                                               \
  add##NAME##Attr(*loop, get<ETY0>(4), get<ETY1>(3), get<ETY2>(2),             \
                  get<ETY3>(1), get<ETY4>(0));                                 \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*loop), get<ETY0>(4));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*loop), get<ETY1>(3));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*loop), get<ETY2>(2));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*loop), get<ETY3>(1));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*loop), get<ETY4>(0));               \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
                                                                               \
  addMetadata(*loop, IRNAME, 5);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*loop));                                     \
  remove##NAME##Attr(*loop);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr6) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
                                                                               \
  add##NAME##Attr(*loop, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                  get<ETY3>(3), get<ETY4>(4), get<ETY5>(5));                   \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*loop), get<ETY0>(0));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*loop), get<ETY1>(1));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*loop), get<ETY2>(2));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*loop), get<ETY3>(3));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*loop), get<ETY4>(4));               \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*loop), get<ETY5>(5));               \
                                                                               \
  add##NAME##Attr(*loop, get<ETY0>(5), get<ETY1>(4), get<ETY2>(3),             \
                  get<ETY3>(2), get<ETY4>(1), get<ETY5>(0));                   \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*loop), get<ETY0>(5));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*loop), get<ETY1>(4));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*loop), get<ETY2>(3));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*loop), get<ETY3>(2));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*loop), get<ETY4>(1));               \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*loop), get<ETY5>(0));               \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
                                                                               \
  addMetadata(*loop, IRNAME, 6);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*loop));                                     \
  remove##NAME##Attr(*loop);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr7) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
                                                                               \
  add##NAME##Attr(*loop, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                  get<ETY3>(3), get<ETY4>(4), get<ETY5>(5), get<ETY6>(6));     \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*loop), get<ETY0>(0));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*loop), get<ETY1>(1));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*loop), get<ETY2>(2));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*loop), get<ETY3>(3));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*loop), get<ETY4>(4));               \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*loop), get<ETY5>(5));               \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(*loop), get<ETY6>(6));               \
                                                                               \
  add##NAME##Attr(*loop, get<ETY0>(6), get<ETY1>(5), get<ETY2>(4),             \
                  get<ETY3>(3), get<ETY4>(2), get<ETY5>(1), get<ETY6>(0));     \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*loop), get<ETY0>(6));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*loop), get<ETY1>(5));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*loop), get<ETY2>(4));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*loop), get<ETY3>(3));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*loop), get<ETY4>(2));               \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*loop), get<ETY5>(1));               \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(*loop), get<ETY6>(0));               \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
                                                                               \
  addMetadata(*loop, IRNAME, 7);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*loop));                                     \
  remove##NAME##Attr(*loop);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr8) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  [[maybe_unused]] Loop *loop = *li.begin();

#define LOOP_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
                                                                               \
  add##NAME##Attr(*loop, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                  get<ETY3>(3), get<ETY4>(4), get<ETY5>(5), get<ETY6>(6),      \
                  get<ETY7>(7));                                               \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*loop), get<ETY0>(0));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*loop), get<ETY1>(1));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*loop), get<ETY2>(2));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*loop), get<ETY3>(3));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*loop), get<ETY4>(4));               \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*loop), get<ETY5>(5));               \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(*loop), get<ETY6>(6));               \
  EXPECT_EQ(get##ENAME7##From##NAME##Attr(*loop), get<ETY7>(7));               \
                                                                               \
  add##NAME##Attr(*loop, get<ETY0>(7), get<ETY1>(6), get<ETY2>(5),             \
                  get<ETY3>(4), get<ETY4>(3), get<ETY5>(2), get<ETY6>(1),      \
                  get<ETY7>(0));                                               \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
  EXPECT_TRUE(has##NAME##Attr(*loop));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*loop), get<ETY0>(7));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*loop), get<ETY1>(6));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*loop), get<ETY2>(5));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*loop), get<ETY3>(4));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*loop), get<ETY4>(3));               \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*loop), get<ETY5>(2));               \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(*loop), get<ETY6>(1));               \
  EXPECT_EQ(get##ENAME7##From##NAME##Attr(*loop), get<ETY7>(0));               \
                                                                               \
  remove##NAME##Attr(*loop);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*loop));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*loop));                                      \
                                                                               \
  addMetadata(*loop, IRNAME, 8);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*loop));                                     \
  remove##NAME##Attr(*loop);

#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

} // namespace
