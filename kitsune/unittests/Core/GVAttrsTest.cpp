//=- GVAttrsTest.cpp - Unit tests for Kitsune's global variable attributes --=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/GVAttrs.h"
#include "TestAttrsCommon.h"
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/GVUtils.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

static void addMetadata(GlobalVariable &g, StringRef attrName,
                        ArrayRef<Metadata *> attrVals) {
  LLVMContext &ctx = g.getContext();
  MDNode *attrList = getAttrList(g);
  MDNode *newAttrList = getAttrListWith(attrName, attrVals, attrList, ctx);

  g.setMetadata(LLVMContext::MD_kit_gv_attrs, newAttrList);
}

// Create metadata consisting of `n` "empty" operands.
static void addMetadata(GlobalVariable &g, StringRef attrName, unsigned n) {
  LLVMContext &ctx = g.getContext();
  MDNode *mdEmpty = MDNode::get(ctx, {});
  SmallVector<Metadata *, 8> attrVals;

  attrVals.append(n, mdEmpty);
  addMetadata(g, attrName, attrVals);
}

TEST(KitGVAttrs, attrName) {
#define GV_ATTR(NAME, IRNAME, ...)                                             \
  EXPECT_EQ(getAttrName(GVAttrKind::NAME), IRNAME);                            \
  EXPECT_TRUE(getAttrName(GVAttrKind::NAME).starts_with("kit.gv."));
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attrKind) {
  EXPECT_EQ(getGVAttrKind("brasenose"), std::nullopt);
#define GV_ATTR(NAME, IRNAME, ...)                                             \
  EXPECT_EQ(getGVAttrKind(IRNAME), GVAttrKind::NAME);
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

#define DECLS(OS, OBJ)                                                         \
  std::string buf;                                                             \
  raw_string_ostream os(buf);                                                  \
  LLVMContext ctx;                                                             \
  Module m("", ctx);                                                           \
  Type *i32 = Type::getInt32Ty(ctx);                                           \
  [[maybe_unused]] GlobalVariable OBJ = m.getOrInsertGlobal("g", i32);

TEST(KitGVAttrs, verifyGeneric) {
  DECLS(os, *g);
#define GV_ATTR_0(NAME, IRNAME, ...)                                           \
  TEST_GENERIC_VERIFY_0(os, *g, GVAttrKind, NAME, IRNAME)
#define GV_ATTR(NAME, IRNAME, ...)                                             \
  TEST_GENERIC_VERIFY_N(os, *g, GVAttrKind, NAME, IRNAME)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attrsGeneric) {
  DECLS(os, *g);

#define GV_ATTR_0(NAME, ...) TEST_GENERIC_ATTR_0(*g, GVAttrKind, NAME)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_0(...)
#define GV_ATTR(NAME, ...) TEST_GENERIC_ATTR_N(*g, GVAttrKind, NAME)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr0) {
  DECLS(os, *g);
#define GV_ATTR_0(...) TEST_ATTR_0(os, *g, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr1) {
  DECLS(os, *g);
#define GV_ATTR_1(...) TEST_ATTR_1(os, *g, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr2) {
  DECLS(os, *g);
#define GV_ATTR_2(...) TEST_ATTR_2(os, *g, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr3) {
  DECLS(os, *g);
#define GV_ATTR_3(...) TEST_ATTR_3(os, *g, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr4) {
  DECLS(os, *g);
#define GV_ATTR_4(...) TEST_ATTR_4(os, *g, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr5) {
  DECLS(os, *g);
#define GV_ATTR_5(...) TEST_ATTR_5(os, *g, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr6) {
  DECLS(os, *g);
#define GV_ATTR_6(...) TEST_ATTR_6(os, *g, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr7) {
  DECLS(os, *g);
#define GV_ATTR_7(...) TEST_ATTR_7(os, *g, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr8) {
  DECLS(os, *g);
#define GV_ATTR_8(...) TEST_ATTR_8(os, *g, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attrLoop) {
  DECLS_LOOP(os, *g, loopF, loopG, lis)
#define GV_ATTR_LOOP(...) TEST_ATTR_LOOP(os, *g, loopF, loopG, lis, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

} // namespace
