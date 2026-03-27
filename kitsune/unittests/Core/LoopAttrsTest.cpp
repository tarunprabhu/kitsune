//===- LoopAttrsTest.cpp - Unit tests for Kitsune's loop attributes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/LoopAttrs.h"
#include "TestAttrsCommon.h"
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Module.h"

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

static void addMetadata(Loop &loop, StringRef attrName,
                        ArrayRef<Metadata *> attrVals) {
  LLVMContext &ctx = getContext(loop);
  MDNode *attrList = getAttrList(loop);
  MDNode *newAttrList = getAttrListWith(attrName, attrVals, attrList, ctx);

  loop.setLoopID(newAttrList);
}

// Create metadata consisting of `n` "empty" operands.
static void addMetadata(Loop &loop, StringRef attrName, unsigned n) {
  LLVMContext &ctx = getContext(loop);
  MDNode *mdEmpty = MDNode::get(ctx, {});
  SmallVector<Metadata *, 8> attrVals;

  attrVals.append(n, mdEmpty);
  addMetadata(loop, attrName, attrVals);
}

TEST(KitLoopAttrs, attrName) {
#define LOOP_ATTR(NAME, IRNAME, ...)                                           \
  EXPECT_EQ(getAttrName(LoopAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(LoopAttrKind::NAME).starts_with("tapir.loop."));
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attrKind) {
  EXPECT_EQ(getLoopAttrKind("wolfson"), std::nullopt);
#define LOOP_ATTR(NAME, IRNAME, ...)                                           \
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
#define LOOP_ATTR_0(NAME, IRNAME, ...)                                         \
  TEST_GENERIC_VERIFY_0(os, *loop, LoopAttrKind, NAME, IRNAME)
#define LOOP_ATTR(NAME, IRNAME, ...)                                           \
  TEST_GENERIC_VERIFY_N(os, *loop, LoopAttrKind, NAME, IRNAME)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attrsGeneric) {
  DECLS(os, *loop);

#define LOOP_ATTR_0(NAME, ...) TEST_GENERIC_ATTR_0(*loop, LoopAttrKind, NAME)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_0(...)
#define LOOP_ATTR(NAME, ...) TEST_GENERIC_ATTR_N(*loop, LoopAttrKind, NAME)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr0) {
  DECLS(os, *loop);
#define LOOP_ATTR_0(...) TEST_ATTR_0(os, *loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr1) {
  DECLS(os, *loop);
#define LOOP_ATTR_1(...) TEST_ATTR_1(os, *loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr2) {
  DECLS(os, *loop);
#define LOOP_ATTR_2(...) TEST_ATTR_2(os, *loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr3) {
  DECLS(os, *loop);
#define LOOP_ATTR_3(...) TEST_ATTR_3(os, *loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr4) {
  DECLS(os, *loop);
#define LOOP_ATTR_4(...) TEST_ATTR_4(os, *loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr5) {
  DECLS(os, *loop);
#define LOOP_ATTR_5(...) TEST_ATTR_5(os, *loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr6) {
  DECLS(os, *loop);
#define LOOP_ATTR_6(...) TEST_ATTR_6(os, *loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr7) {
  DECLS(os, *loop);
#define LOOP_ATTR_7(...) TEST_ATTR_7(os, *loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attr8) {
  DECLS(os, *loop);
#define LOOP_ATTR_8(...) TEST_ATTR_8(os, *loop, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST(KitLoopAttrs, attrLoop) {
  DECLS_LOOP(os, *loop, loopF, loopG, lis)
#define LOOP_ATTR_LOOP(...)                                                    \
  TEST_ATTR_LOOP(os, *loop, loopF, loopG, lis, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

} // namespace
