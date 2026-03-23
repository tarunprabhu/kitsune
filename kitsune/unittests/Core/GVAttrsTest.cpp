//=- GVAttrsTest.cpp - Unit tests for Kitsune's global variable attributes --=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/GVAttrs.h"
#include "TestValues.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

// The standard accessors do not allow us to create invalid attributes. To
// create one, we have to know how these are added to the function. This is not
// unreasonable since the create functions are a fairly thin wrappers around
// LLVM's existing support.
static void addMetadata(GlobalVariable &g, StringRef name,
                        ArrayRef<Metadata *> ops) {
  LLVMContext &ctx = g.getContext();
  g.addMetadata(name, *MDNode::get(ctx, ops));
}

// Create metadata consisting of `n` "empty" operands.
static void addMetadata(GlobalVariable &g, StringRef name, unsigned n) {
  LLVMContext &ctx = g.getContext();
  MDNode *mdEmpty = MDNode::get(ctx, {});
  SmallVector<Metadata *, 8> ops;

  ops.append(n, mdEmpty);
  addMetadata(g, name, ops);
}

TEST(KitGVAttrs, attrName) {
#define GV_ATTR(NAME, IRNAME, TYPE)                                            \
  EXPECT_EQ(getAttrName(GVAttrKind::NAME), IRNAME);                            \
  EXPECT_TRUE(getAttrName(GVAttrKind::NAME).starts_with("kit.gv."));
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attrKind) {
  EXPECT_EQ(getGVAttrKind("brasenose"), std::nullopt);

#define GV_ATTR(NAME, IRNAME, TYPE)                                            \
  EXPECT_EQ(getGVAttrKind(IRNAME), GVAttrKind::NAME);
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, verify) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_0(NAME, IRNAME)                                                \
  addMetadata(g, IRNAME, MDString::get(ctx, ""));                              \
                                                                               \
  EXPECT_FALSE(verifyAttr(g, GVAttrKind::NAME));                               \
  EXPECT_FALSE(verify##NAME##Attr(g));                                         \
                                                                               \
  remove##NAME##Attr(g);

#define GV_ATTR(NAME, IRNAME, TYPE)                                            \
  addMetadata(g, IRNAME, {});                                                  \
                                                                               \
  EXPECT_FALSE(verifyAttr(g, GVAttrKind::NAME));                               \
  EXPECT_FALSE(verify##NAME##Attr(g));                                         \
                                                                               \
  remove##NAME##Attr(g);

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attrsGeneric) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_0(NAME, IRNAME)                                                \
  EXPECT_FALSE(hasAttr(g, GVAttrKind::NAME));                                  \
  addAttr(GV, GVAttrKind::NAME);                                               \
  EXPECT_TRUE(hasAttr(g, GVAttrKind::NAME));                                   \
  addAttr(GV, GVAttrKind::NAME);                                               \
  EXPECT_TRUE(hasAttr(g, GVAttrKind::NAME));                                   \
  removeAttr(GV, GVAttrKind::NAME);                                            \
  EXPECT_FALSE(hasAttr(g, GVAttrKind::NAME));
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_0(NAME, IRNAME)
#define GV_ATTR(NAME, IRNAME, TYPE)                                            \
  EXPECT_EXIT(addAttr(g, GVAttrKind::NAME), ::testing::ExitedWithCode(1),      \
              "error: cannot add attribute");
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr0) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_0(NAME, IRNAME)                                                \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
                                                                               \
  add##NAME##Attr(g);                                                          \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
                                                                               \
  add##NAME##Attr(g);                                                          \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
                                                                               \
  remove##NAME##Attr(g);                                                       \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
                                                                               \
  addMetadata(g, IRNAME, 0);                                                   \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  remove##NAME##Attr(g);

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr1) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_1(NAME, IRNAME, TYPE)                                          \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
                                                                               \
  add##NAME##Attr(g, get<TYPE>(0));                                            \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##NAME##Attr(g), get<TYPE>(0));                                 \
                                                                               \
  add##NAME##Attr(g, get<TYPE>(1));                                            \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##NAME##Attr(g), get<TYPE>(1));                                 \
                                                                               \
  remove##NAME##Attr(g);                                                       \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
                                                                               \
  addMetadata(g, IRNAME, 1);                                                   \
  EXPECT_FALSE(verify##NAME##Attr(g));                                         \
  remove##NAME##Attr(g);

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr2) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)          \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
                                                                               \
  add##NAME##Attr(g, get<ETY0>(0), get<ETY1>(1));                              \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(g), get<ETY0>(0));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(g), get<ETY1>(1));                   \
                                                                               \
  add##NAME##Attr(g, get<ETY0>(1), get<ETY1>(0));                              \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(g), get<ETY0>(1));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(g), get<ETY1>(0));                   \
                                                                               \
  remove##NAME##Attr(g);                                                       \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
                                                                               \
  addMetadata(g, IRNAME, 2);                                                   \
  EXPECT_FALSE(verify##NAME##Attr(g));                                         \
  remove##NAME##Attr(g);

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr3) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2)                                                 \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
                                                                               \
  add##NAME##Attr(g, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2));                \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(g), get<ETY0>(0));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(g), get<ETY1>(1));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(g), get<ETY2>(2));                   \
                                                                               \
  add##NAME##Attr(g, get<ETY0>(2), get<ETY1>(1), get<ETY2>(0));                \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(g), get<ETY0>(2));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(g), get<ETY1>(1));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(g), get<ETY2>(0));                   \
                                                                               \
  remove##NAME##Attr(g);                                                       \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
                                                                               \
  addMetadata(g, IRNAME, 3);                                                   \
  EXPECT_FALSE(verify##NAME##Attr(g));                                         \
  remove##NAME##Attr(g);

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr4) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3)                              \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
                                                                               \
  add##NAME##Attr(g, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3));  \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(g), get<ETY0>(0));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(g), get<ETY1>(1));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(g), get<ETY2>(2));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(g), get<ETY3>(3));                   \
                                                                               \
  add##NAME##Attr(g, get<ETY0>(3), get<ETY1>(2), get<ETY2>(1), get<ETY3>(0));  \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(g), get<ETY0>(3));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(g), get<ETY1>(2));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(g), get<ETY2>(1));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(g), get<ETY3>(0));                   \
                                                                               \
  remove##NAME##Attr(g);                                                       \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
                                                                               \
  addMetadata(g, IRNAME, 4);                                                   \
  EXPECT_FALSE(verify##NAME##Attr(g));                                         \
  remove##NAME##Attr(g);

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr5) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)           \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
                                                                               \
  add##NAME##Attr(g, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3),   \
                  get<ETY4>(4));                                               \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(g), get<ETY0>(0));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(g), get<ETY1>(1));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(g), get<ETY2>(2));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(g), get<ETY3>(3));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(g), get<ETY4>(4));                   \
                                                                               \
  add##NAME##Attr(g, get<ETY0>(4), get<ETY1>(3), get<ETY2>(2), get<ETY3>(1),   \
                  get<ETY4>(0));                                               \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(g), get<ETY0>(4));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(g), get<ETY1>(3));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(g), get<ETY2>(2));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(g), get<ETY3>(1));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(g), get<ETY4>(0));                   \
                                                                               \
  remove##NAME##Attr(g);                                                       \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
                                                                               \
  addMetadata(g, IRNAME, 5);                                                   \
  EXPECT_FALSE(verify##NAME##Attr(g));                                         \
  remove##NAME##Attr(g);

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr6) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,     \
                  ENAME5, EN5)                                                 \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
                                                                               \
  add##NAME##Attr(g, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3),   \
                  get<ETY4>(4), get<ETY5>(5));                                 \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(g), get<ETY0>(0));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(g), get<ETY1>(1));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(g), get<ETY2>(2));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(g), get<ETY3>(3));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(g), get<ETY4>(4));                   \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(g), get<ETY5>(5));                   \
                                                                               \
  add##NAME##Attr(g, get<ETY0>(5), get<ETY1>(4), get<ETY2>(3), get<ETY3>(2),   \
                  get<ETY4>(1), get<ETY5>(0));                                 \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(g), get<ETY0>(5));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(g), get<ETY1>(4));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(g), get<ETY2>(3));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(g), get<ETY3>(2));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(g), get<ETY4>(1));                   \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(g), get<ETY5>(0));                   \
                                                                               \
  remove##NAME##Attr(g);                                                       \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
                                                                               \
  addMetadata(g, IRNAME, 6);                                                   \
  EXPECT_FALSE(verify##NAME##Attr(g));                                         \
  remove##NAME##Attr(g);

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr7) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,     \
                  ENAME5, EN5, ETY6, ENAME6, EN6)                              \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
                                                                               \
  add##NAME##Attr(g, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3),   \
                  get<ETY4>(4), get<ETY5>(5), get<ETY6>(6));                   \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(g), get<ETY0>(0));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(g), get<ETY1>(1));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(g), get<ETY2>(2));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(g), get<ETY3>(3));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(g), get<ETY4>(4));                   \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(g), get<ETY5>(5));                   \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(g), get<ETY6>(6));                   \
                                                                               \
  add##NAME##Attr(g, get<ETY0>(6), get<ETY1>(5), get<ETY2>(4), get<ETY3>(3),   \
                  get<ETY4>(2), get<ETY5>(1), get<ETY6>(0));                   \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(g), get<ETY0>(6));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(g), get<ETY1>(5));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(g), get<ETY2>(4));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(g), get<ETY3>(3));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(g), get<ETY4>(2));                   \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(g), get<ETY5>(1));                   \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(g), get<ETY6>(0));                   \
                                                                               \
  remove##NAME##Attr(g);                                                       \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
                                                                               \
  addMetadata(g, IRNAME, 7);                                                   \
  EXPECT_FALSE(verify##NAME##Attr(g));                                         \
  remove##NAME##Attr(g);

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr8) {
  LLVMContext ctx;
  Type *i32 = Type::getInt32Ty(ctx);
  [[maybe_unused]] GlobalVariable g(i32, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage);

#define GV_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,    \
                  ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,     \
                  ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)           \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
                                                                               \
  add##NAME##Attr(g, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3),   \
                  get<ETY4>(4), get<ETY5>(5), get<ETY6>(6), get<ETY7>(7));     \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(g), get<ETY0>(0));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(g), get<ETY1>(1));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(g), get<ETY2>(2));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(g), get<ETY3>(3));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(g), get<ETY4>(4));                   \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(g), get<ETY5>(5));                   \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(g), get<ETY6>(6));                   \
  EXPECT_EQ(get##ENAME7##From##NAME##Attr(g), get<ETY7>(7));                   \
                                                                               \
  add##NAME##Attr(g, get<ETY0>(7), get<ETY1>(6), get<ETY2>(5), get<ETY3>(4),   \
                  get<ETY4>(3), get<ETY5>(2), get<ETY6>(1), get<ETY7>(0));     \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
  EXPECT_TRUE(has##NAME##Attr(g));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(g), get<ETY0>(7));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(g), get<ETY1>(6));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(g), get<ETY2>(5));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(g), get<ETY3>(4));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(g), get<ETY4>(3));                   \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(g), get<ETY5>(2));                   \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(g), get<ETY6>(1));                   \
  EXPECT_EQ(get##ENAME7##From##NAME##Attr(g), get<ETY7>(0));                   \
                                                                               \
  remove##NAME##Attr(g);                                                       \
  EXPECT_FALSE(has##NAME##Attr(g));                                            \
  EXPECT_TRUE(verify##NAME##Attr(g));                                          \
                                                                               \
  addMetadata(g, IRNAME, 8);                                                   \
  EXPECT_FALSE(verify##NAME##Attr(g));                                         \
  remove##NAME##Attr(g);

#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

} // namespace
