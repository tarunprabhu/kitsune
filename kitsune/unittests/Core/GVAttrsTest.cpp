//=- GVAttrsTest.cpp - Unit tests for Kitsune's global variable attributes --=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/GVAttrs.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitGVAttrs, attrName) {
#define GV_ATTR(NAME, TYPE, IRNAME)                                            \
  EXPECT_EQ(getAttrName(GVAttrKind::NAME), IRNAME);                            \
  EXPECT_TRUE(getAttrName(GVAttrKind::NAME).starts_with("kit.gv."));
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attrKind) {
  EXPECT_EQ(getGVAttrKind("whoops"), std::nullopt);

#define GV_ATTR(NAME, TYPE, IRNAME)                                            \
  EXPECT_EQ(getGVAttrKind(IRNAME), GVAttrKind::NAME);
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

#define CHECK_GENERIC_FLAG(NAME, GV)                                           \
  EXPECT_FALSE(hasAttr(GV, GVAttrKind::NAME));                                 \
  addAttr(GV, GVAttrKind::NAME);                                               \
  EXPECT_TRUE(hasAttr(GV, GVAttrKind::NAME));                                  \
  removeAttr(GV, GVAttrKind::NAME);                                            \
  EXPECT_FALSE(hasAttr(GV, GVAttrKind::NAME));

#define CHECK_GENERIC(NAME, GV, VAL)                                           \
  EXPECT_FALSE(hasAttr(GV, GVAttrKind::NAME));                                 \
  add##NAME##Attr(GV, VAL);                                                    \
  EXPECT_TRUE(hasAttr(GV, GVAttrKind::NAME));                                  \
  removeAttr(GV, GVAttrKind::NAME);                                            \
  EXPECT_FALSE(hasAttr(GV, GVAttrKind::NAME));                                 \
  EXPECT_EXIT(addAttr(GV, GVAttrKind::NAME), ::testing::ExitedWithCode(1),     \
              "error: cannot add attribute");

TEST(KitGVAttrs, attrsGeneric) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_FLAG(NAME, IRNAME) CHECK_GENERIC_FLAG(NAME, g)
#define GV_ATTR_ENUM(NAME, IRNAME, TYPE) CHECK_GENERIC(NAME, g, (TYPE)1)
#define GV_ATTR_F32(NAME, IRNAME) CHECK_GENERIC(NAME, g, 3.14F)
#define GV_ATTR_F64(NAME, IRNAME) CHECK_GENERIC(NAME, g, 3.141592653)
#define GV_ATTR_I32(NAME, IRNAME) CHECK_GENERIC(NAME, g, 277)
#define GV_ATTR_I64(NAME, IRNAME) CHECK_GENERIC(NAME, g, 409L)
#define GV_ATTR_STR(NAME, IRNAME) CHECK_GENERIC(NAME, g, "edinburgh")
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, flagAttrs) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_FLAG(NAME, IRNAME)                                             \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
                                                                               \
  add##NAME##Attr(g);                                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
                                                                               \
  add##NAME##Attr(g);                                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
                                                                               \
  remove##NAME##Attr(g);                                                       \
  EXPECT_FALSE(has##NAME##Attr(g));

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

#define CHECK_ACCESSORS(NAME, GV, VAL1, VAL2)                                  \
  EXPECT_FALSE(has##NAME##Attr(GV));                                           \
                                                                               \
  add##NAME##Attr(GV, VAL1);                                                   \
  EXPECT_TRUE(has##NAME##Attr(GV));                                            \
  EXPECT_EQ(get##NAME##Attr(GV), (VAL1));                                      \
  llvm::errs() << g << "\n";                                                   \
                                                                               \
  add##NAME##Attr(GV, VAL2);                                                   \
  EXPECT_TRUE(has##NAME##Attr(GV));                                            \
  EXPECT_EQ(get##NAME##Attr(GV), (VAL2));                                      \
                                                                               \
  remove##NAME##Attr(GV);                                                      \
  EXPECT_FALSE(has##NAME##Attr(GV));

TEST(KitGVAttrs, enumAttrs) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

  // WARNING: This is somewhat risky because there is no guarantee that the
  // integer values 1 and 2 will be valid for every enum type that we may have
  // an attribute for. If this ever happens, change this test. It may be
  // sufficient to just check for some enum-valued attribute instead of all of
  // them.
#define GV_ATTR_ENUM(NAME, IRNAME, TYPE)                                       \
  CHECK_ACCESSORS(NAME, g, (TYPE)1, (TYPE)2)

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, f32Test) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_F32(NAME, IRNAME) CHECK_ACCESSORS(NAME, g, 3.14159F, 2.71828F)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, f64Test) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_F64(NAME, IRNAME)                                              \
  CHECK_ACCESSORS(NAME, g, 3.1415926535, 2.71828)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, i32Test) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_I32(NAME, IRNAME) CHECK_ACCESSORS(NAME, g, 151, 157)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, i64Test) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_I64(NAME, IRNAME) CHECK_ACCESSORS(NAME, g, 131L, 137L)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, strTest) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_STR(NAME, IRNAME) CHECK_ACCESSORS(NAME, g, "balliol", "keble")
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

} // namespace
