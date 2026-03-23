//==- InstAttrsTest.cpp - Unit tests for Kitsune's instruction attributes --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/InstAttrs.h"
#include "TestValues.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

// The standard accessors do not allow us to create invalid attributes. To
// create one, we have to know how these are added to the function. This is not
// unreasonable since the create functions are a fairly thin wrappers around
// LLVM's existing support.
static void addMetadata(Instruction &inst, StringRef name,
                        ArrayRef<Metadata *> ops) {
  LLVMContext &ctx = inst.getContext();
  inst.setMetadata(name, MDNode::get(ctx, ops));
}

// Create metadata consisting of `n` "empty" operands.
static void addMetadata(Instruction &inst, StringRef name, unsigned n) {
  LLVMContext &ctx = inst.getContext();
  MDNode *mdEmpty = MDNode::get(ctx, {});
  SmallVector<Metadata *, 8> ops;

  ops.append(n, mdEmpty);
  addMetadata(inst, name, ops);
}

static constexpr StringRef ll = R"(
define void @f(i64 %n) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.header ]
  %inc.i = add i64 %i, 1
  %exitcond.i.not = icmp eq i64 %inc.i, %n
  br i1 %exitcond.i.not, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  ret void
}

define void @g(i64 %n) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.header ]
  %inc.i = add i64 %i, 1
  %exitcond.i.not = icmp eq i64 %inc.i, %n
  br i1 %exitcond.i.not, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  ret void
}

!0 = distinct !{!0}
!1 = distinct !{!1}
)";

static std::unique_ptr<Module> parseIR(LLVMContext &ctx, StringRef ir) {
  SMDiagnostic err;
  std::unique_ptr<Module> m = parseAssemblyString(ir, err, ctx);
  if (!m)
    err.print("parseIR", errs());
  return m;
}

TEST(KitInstAttrs, attrName) {
#define INST_ATTR(NAME, IRNAME, TYPE)                                          \
  EXPECT_EQ(getAttrName(InstAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(InstAttrKind::NAME).starts_with("kit.inst."));
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attrKind) {
  EXPECT_EQ(getInstAttrKind("queen's"), std::nullopt);

#define INST_ATTR(NAME, IRNAME, TYPE)                                          \
  EXPECT_EQ(getInstAttrKind(IRNAME), InstAttrKind::NAME);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, verify) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_0(NAME, IRNAME)                                              \
  addMetadata(*inst, IRNAME, MDString::get(ctx, ""));                          \
                                                                               \
  EXPECT_FALSE(verifyAttr(*inst, InstAttrKind::NAME));                         \
  EXPECT_FALSE(verify##NAME##Attr(*inst));                                     \
                                                                               \
  remove##NAME##Attr(*inst);

#define INST_ATTR(NAME, IRNAME, TYPE)                                          \
  addMetadata(*inst, IRNAME, {});                                              \
                                                                               \
  EXPECT_FALSE(verifyAttr(*inst, InstAttrKind::NAME));                         \
  EXPECT_FALSE(verify##NAME##Attr(*inst));                                     \
                                                                               \
  remove##NAME##Attr(*inst);

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attrsGeneric) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_0(NAME, IRNAME)                                              \
  EXPECT_FALSE(hasAttr(*inst, InstAttrKind::NAME));                            \
  addAttr(*inst, InstAttrKind::NAME);                                          \
  EXPECT_TRUE(hasAttr(*inst, InstAttrKind::NAME));                             \
  addAttr(*inst, InstAttrKind::NAME);                                          \
  EXPECT_TRUE(hasAttr(*inst, InstAttrKind::NAME));                             \
  removeAttr(*inst, InstAttrKind::NAME);                                       \
  EXPECT_FALSE(hasAttr(*inst, InstAttrKind::NAME));
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_0(NAME, IRNAME)
#define INST_ATTR(NAME, IRNAME, TYPE)                                          \
  EXPECT_EXIT(addAttr(*inst, InstAttrKind::NAME),                              \
              ::testing::ExitedWithCode(1), "error: cannot add attribute");
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr0) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_0(NAME, IRNAME)                                              \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
                                                                               \
  add##NAME##Attr(*inst);                                                      \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
                                                                               \
  add##NAME##Attr(*inst);                                                      \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
                                                                               \
  remove##NAME##Attr(*inst);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
                                                                               \
  addMetadata(*inst, IRNAME, 0);                                               \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  remove##NAME##Attr(*inst);

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr1) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_1(NAME, IRNAME, TYPE)                                        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
                                                                               \
  add##NAME##Attr(*inst, get<TYPE>(0));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##NAME##Attr(*inst), get<TYPE>(0));                             \
                                                                               \
  add##NAME##Attr(*inst, get<TYPE>(1));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##NAME##Attr(*inst), get<TYPE>(1));                             \
                                                                               \
  remove##NAME##Attr(*inst);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
                                                                               \
  addMetadata(*inst, IRNAME, 1);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*inst));                                     \
  remove##NAME##Attr(*inst);

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, loop) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dtf(*f);
  LoopInfo lif(dtf);
  Function *g = m->getFunction("g");
  DominatorTree dtg(*g);
  LoopInfo lig(dtg);
  SmallVector<const LoopInfo *, 4> lis = {&lig, &lif};
  [[maybe_unused]] Loop *loopF = *lif.begin();
  [[maybe_unused]] Loop *loopG = *lig.begin();
  [[maybe_unused]] Instruction *inst = ReturnInst::Create(ctx);

#define INST_ATTR_LOOP(NAME, IRNAME)                                           \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
                                                                               \
  add##NAME##Attr(*inst, loopF);                                               \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##NAME##Attr(*inst, lis), loopF);                               \
                                                                               \
  add##NAME##Attr(*inst, *loopG);                                              \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##NAME##Attr(*inst, lis), loopG);                               \
                                                                               \
  remove##NAME##Attr(*inst);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
                                                                               \
  addMetadata(*inst, IRNAME, 1);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*inst));                                     \
  remove##NAME##Attr(*inst);

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr2) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
                                                                               \
  add##NAME##Attr(*inst, get<ETY0>(0), get<ETY1>(1));                          \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*inst), get<ETY0>(0));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*inst), get<ETY1>(1));               \
                                                                               \
  add##NAME##Attr(*inst, get<ETY0>(1), get<ETY1>(0));                          \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*inst), get<ETY0>(1));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*inst), get<ETY1>(0));               \
                                                                               \
  remove##NAME##Attr(*inst);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
                                                                               \
  addMetadata(*inst, IRNAME, 2);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*inst));                                     \
  remove##NAME##Attr(*inst);

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr3) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
                                                                               \
  add##NAME##Attr(*inst, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2));            \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*inst), get<ETY0>(0));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*inst), get<ETY1>(1));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*inst), get<ETY2>(2));               \
                                                                               \
  add##NAME##Attr(*inst, get<ETY0>(2), get<ETY1>(1), get<ETY2>(0));            \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*inst), get<ETY0>(2));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*inst), get<ETY1>(1));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*inst), get<ETY2>(0));               \
                                                                               \
  remove##NAME##Attr(*inst);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
                                                                               \
  addMetadata(*inst, IRNAME, 3);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*inst));                                     \
  remove##NAME##Attr(*inst);

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr4) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
                                                                               \
  add##NAME##Attr(*inst, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                  get<ETY3>(3));                                               \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*inst), get<ETY0>(0));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*inst), get<ETY1>(1));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*inst), get<ETY2>(2));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*inst), get<ETY3>(3));               \
                                                                               \
  add##NAME##Attr(*inst, get<ETY0>(3), get<ETY1>(2), get<ETY2>(1),             \
                  get<ETY3>(0));                                               \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*inst), get<ETY0>(3));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*inst), get<ETY1>(2));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*inst), get<ETY2>(1));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*inst), get<ETY3>(0));               \
                                                                               \
  remove##NAME##Attr(*inst);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
                                                                               \
  addMetadata(*inst, IRNAME, 4);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*inst));                                     \
  remove##NAME##Attr(*inst);

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr5) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
                                                                               \
  add##NAME##Attr(*inst, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                  get<ETY3>(3), get<ETY4>(4));                                 \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*inst), get<ETY0>(0));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*inst), get<ETY1>(1));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*inst), get<ETY2>(2));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*inst), get<ETY3>(3));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*inst), get<ETY4>(4));               \
                                                                               \
  add##NAME##Attr(*inst, get<ETY0>(4), get<ETY1>(3), get<ETY2>(2),             \
                  get<ETY3>(1), get<ETY4>(0));                                 \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*inst), get<ETY0>(4));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*inst), get<ETY1>(3));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*inst), get<ETY2>(2));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*inst), get<ETY3>(1));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*inst), get<ETY4>(0));               \
                                                                               \
  remove##NAME##Attr(*inst);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
                                                                               \
  addMetadata(*inst, IRNAME, 5);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*inst));                                     \
  remove##NAME##Attr(*inst);

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr6) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
                                                                               \
  add##NAME##Attr(*inst, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                  get<ETY3>(3), get<ETY4>(4), get<ETY5>(5));                   \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*inst), get<ETY0>(0));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*inst), get<ETY1>(1));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*inst), get<ETY2>(2));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*inst), get<ETY3>(3));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*inst), get<ETY4>(4));               \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*inst), get<ETY5>(5));               \
                                                                               \
  add##NAME##Attr(*inst, get<ETY0>(5), get<ETY1>(4), get<ETY2>(3),             \
                  get<ETY3>(2), get<ETY4>(1), get<ETY5>(0));                   \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*inst), get<ETY0>(5));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*inst), get<ETY1>(4));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*inst), get<ETY2>(3));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*inst), get<ETY3>(2));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*inst), get<ETY4>(1));               \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*inst), get<ETY5>(0));               \
                                                                               \
  remove##NAME##Attr(*inst);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
                                                                               \
  addMetadata(*inst, IRNAME, 6);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*inst));                                     \
  remove##NAME##Attr(*inst);

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr7) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
                                                                               \
  add##NAME##Attr(*inst, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                  get<ETY3>(3), get<ETY4>(4), get<ETY5>(5), get<ETY6>(6));     \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*inst), get<ETY0>(0));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*inst), get<ETY1>(1));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*inst), get<ETY2>(2));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*inst), get<ETY3>(3));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*inst), get<ETY4>(4));               \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*inst), get<ETY5>(5));               \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(*inst), get<ETY6>(6));               \
                                                                               \
  add##NAME##Attr(*inst, get<ETY0>(6), get<ETY1>(5), get<ETY2>(4),             \
                  get<ETY3>(3), get<ETY4>(2), get<ETY5>(1), get<ETY6>(0));     \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*inst), get<ETY0>(6));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*inst), get<ETY1>(5));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*inst), get<ETY2>(4));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*inst), get<ETY3>(3));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*inst), get<ETY4>(2));               \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*inst), get<ETY5>(1));               \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(*inst), get<ETY6>(0));               \
                                                                               \
  remove##NAME##Attr(*inst);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
                                                                               \
  addMetadata(*inst, IRNAME, 7);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*inst));                                     \
  remove##NAME##Attr(*inst);

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr8) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
                                                                               \
  add##NAME##Attr(*inst, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                  get<ETY3>(3), get<ETY4>(4), get<ETY5>(5), get<ETY6>(6),      \
                  get<ETY7>(7));                                               \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*inst), get<ETY0>(0));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*inst), get<ETY1>(1));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*inst), get<ETY2>(2));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*inst), get<ETY3>(3));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*inst), get<ETY4>(4));               \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*inst), get<ETY5>(5));               \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(*inst), get<ETY6>(6));               \
  EXPECT_EQ(get##ENAME7##From##NAME##Attr(*inst), get<ETY7>(7));               \
                                                                               \
  add##NAME##Attr(*inst, get<ETY0>(7), get<ETY1>(6), get<ETY2>(5),             \
                  get<ETY3>(4), get<ETY4>(3), get<ETY5>(2), get<ETY6>(1),      \
                  get<ETY7>(0));                                               \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*inst), get<ETY0>(7));               \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*inst), get<ETY1>(6));               \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*inst), get<ETY2>(5));               \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*inst), get<ETY3>(4));               \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*inst), get<ETY4>(3));               \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*inst), get<ETY5>(2));               \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(*inst), get<ETY6>(1));               \
  EXPECT_EQ(get##ENAME7##From##NAME##Attr(*inst), get<ETY7>(0));               \
                                                                               \
  remove##NAME##Attr(*inst);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
                                                                               \
  addMetadata(*inst, IRNAME, 8);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*inst));                                     \
  remove##NAME##Attr(*inst);

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

} // namespace
