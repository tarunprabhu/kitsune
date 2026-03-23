//===- ModuleAttrsTest.cpp - Unit tests for Kitsune's module attributes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ModuleAttrs.h"
#include "TestValues.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitModuleAttrs, attrName) {
#define MODULE_ATTR(NAME, IRNAME, TYPE)                                        \
  EXPECT_EQ(getAttrName(ModuleAttrKind::NAME), IRNAME);                        \
  EXPECT_TRUE(getAttrName(ModuleAttrKind::NAME).starts_with("kit.module."));
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attrKind) {
  EXPECT_EQ(getModuleAttrKind("balliol"), std::nullopt);

#define MODULE_ATTR(NAME, IRNAME, TYPE)                         \
  EXPECT_EQ(getModuleAttrKind(IRNAME), ModuleAttrKind::NAME);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attrsGeneric) {
  LLVMContext ctx;
  Module m("", ctx);

#define MODULE_ATTR_0(NAME, IRNAME)                                            \
  EXPECT_FALSE(hasAttr(m, ModuleAttrKind::NAME));                              \
  addAttr(m, ModuleAttrKind::NAME);                                            \
  EXPECT_TRUE(hasAttr(m, ModuleAttrKind::NAME));                               \
  addAttr(m, ModuleAttrKind::NAME);                                            \
  EXPECT_TRUE(hasAttr(m, ModuleAttrKind::NAME));                               \
  removeAttr(m, ModuleAttrKind::NAME);                                         \
  EXPECT_FALSE(hasAttr(m, ModuleAttrKind::NAME));
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_0(NAME, IRNAME)
#define MODULE_ATTR(NAME, IRNAME, TYPE)                                        \
  EXPECT_EXIT(addAttr(m, ModuleAttrKind::NAME), ::testing::ExitedWithCode(1),  \
              "error: cannot add attribute");
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr0) {
  LLVMContext ctx;
  Module m("", ctx);

#define MODULE_ATTR_0(NAME, IRNAME)                                            \
  EXPECT_FALSE(has##NAME##Attr(m));                                            \
                                                                               \
  add##NAME##Attr(m);                                                          \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
                                                                               \
  add##NAME##Attr(m);                                                          \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
                                                                               \
  remove##NAME##Attr(m);                                                       \
  EXPECT_FALSE(has##NAME##Attr(m));

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr1) {
  LLVMContext ctx;
  Module m("", ctx);

#define MODULE_ATTR_1(NAME, IRNAME, TYPE)                                      \
  EXPECT_FALSE(has##NAME##Attr(m));                                            \
                                                                               \
  add##NAME##Attr(m, get<TYPE>(0));                                            \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##NAME##Attr(m), get<TYPE>(0));                                 \
                                                                               \
  add##NAME##Attr(m, get<TYPE>(1));                                            \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##NAME##Attr(m), get<TYPE>(1));                                 \
                                                                               \
  remove##NAME##Attr(m);                                                       \
  EXPECT_FALSE(has##NAME##Attr(m));

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr2) {
  LLVMContext ctx;
  Module m("", ctx);

#define MODULE_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)      \
  EXPECT_FALSE(has##NAME##Attr(m));                                            \
                                                                               \
  add##NAME##Attr(m, get<ETY0>(0), get<ETY1>(1));                              \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(m), get<ETY0>(0));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(m), get<ETY1>(1));                   \
                                                                               \
  add##NAME##Attr(m, get<ETY0>(1), get<ETY1>(0));                              \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(m), get<ETY0>(1));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(m), get<ETY1>(0));                   \
                                                                               \
  remove##NAME##Attr(m);                                                       \
  EXPECT_FALSE(has##NAME##Attr(m));

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr3) {
  LLVMContext ctx;
  Module m("", ctx);

#define MODULE_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2)                                       \
  EXPECT_FALSE(has##NAME##Attr(m));                                            \
                                                                               \
  add##NAME##Attr(m, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2));                \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(m), get<ETY0>(0));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(m), get<ETY1>(1));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(m), get<ETY2>(2));                   \
                                                                               \
  add##NAME##Attr(m, get<ETY0>(2), get<ETY1>(1), get<ETY2>(0));                \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(m), get<ETY0>(2));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(m), get<ETY1>(1));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(m), get<ETY2>(0));                   \
                                                                               \
  remove##NAME##Attr(m);                                                       \
  EXPECT_FALSE(has##NAME##Attr(m));

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr4) {
  LLVMContext ctx;
  Module m("", ctx);

#define MODULE_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)                    \
  EXPECT_FALSE(has##NAME##Attr(m));                                            \
                                                                               \
  add##NAME##Attr(m, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3));  \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(m), get<ETY0>(0));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(m), get<ETY1>(1));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(m), get<ETY2>(2));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(m), get<ETY3>(3));                   \
                                                                               \
  add##NAME##Attr(m, get<ETY0>(3), get<ETY1>(2), get<ETY2>(1), get<ETY3>(0));  \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(m), get<ETY0>(3));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(m), get<ETY1>(2));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(m), get<ETY2>(1));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(m), get<ETY3>(0));                   \
                                                                               \
  remove##NAME##Attr(m);                                                       \
  EXPECT_FALSE(has##NAME##Attr(m));

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr5) {
  LLVMContext ctx;
  Module m("", ctx);

#define MODULE_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4) \
  EXPECT_FALSE(has##NAME##Attr(m));                                            \
                                                                               \
  add##NAME##Attr(m, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3),   \
                  get<ETY4>(4));                                               \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(m), get<ETY0>(0));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(m), get<ETY1>(1));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(m), get<ETY2>(2));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(m), get<ETY3>(3));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(m), get<ETY4>(4));                   \
                                                                               \
  add##NAME##Attr(m, get<ETY0>(4), get<ETY1>(3), get<ETY2>(2), get<ETY3>(1),   \
                  get<ETY4>(0));                                               \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(m), get<ETY0>(4));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(m), get<ETY1>(3));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(m), get<ETY2>(2));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(m), get<ETY3>(1));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(m), get<ETY4>(0));                   \
                                                                               \
  remove##NAME##Attr(m);                                                       \
  EXPECT_FALSE(has##NAME##Attr(m));

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr6) {
  LLVMContext ctx;
  Module m("", ctx);

#define MODULE_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5)                                       \
  EXPECT_FALSE(has##NAME##Attr(m));                                            \
                                                                               \
  add##NAME##Attr(m, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3),   \
                  get<ETY4>(4), get<ETY5>(5));                                 \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(m), get<ETY0>(0));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(m), get<ETY1>(1));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(m), get<ETY2>(2));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(m), get<ETY3>(3));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(m), get<ETY4>(4));                   \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(m), get<ETY5>(5));                   \
                                                                               \
  add##NAME##Attr(m, get<ETY0>(5), get<ETY1>(4), get<ETY2>(3), get<ETY3>(2),   \
                  get<ETY4>(1), get<ETY5>(0));                                 \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(m), get<ETY0>(5));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(m), get<ETY1>(4));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(m), get<ETY2>(3));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(m), get<ETY3>(2));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(m), get<ETY4>(1));                   \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(m), get<ETY5>(0));                   \
                                                                               \
  remove##NAME##Attr(m);                                                       \
  EXPECT_FALSE(has##NAME##Attr(m));

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr7) {
  LLVMContext ctx;
  Module m("", ctx);

#define MODULE_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5, ETY6, ENAME6, EN6)                    \
  EXPECT_FALSE(has##NAME##Attr(m));                                            \
                                                                               \
  add##NAME##Attr(m, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3),   \
                  get<ETY4>(4), get<ETY5>(5), get<ETY6>(6));                   \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(m), get<ETY0>(0));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(m), get<ETY1>(1));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(m), get<ETY2>(2));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(m), get<ETY3>(3));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(m), get<ETY4>(4));                   \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(m), get<ETY5>(5));                   \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(m), get<ETY6>(6));                   \
                                                                               \
  add##NAME##Attr(m, get<ETY0>(6), get<ETY1>(5), get<ETY2>(4), get<ETY3>(3),   \
                  get<ETY4>(2), get<ETY5>(1), get<ETY6>(0));                   \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(m), get<ETY0>(6));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(m), get<ETY1>(5));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(m), get<ETY2>(4));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(m), get<ETY3>(3));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(m), get<ETY4>(2));                   \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(m), get<ETY5>(1));                   \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(m), get<ETY6>(0));                   \
                                                                               \
  remove##NAME##Attr(m);                                                       \
  EXPECT_FALSE(has##NAME##Attr(m));

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr8) {
  LLVMContext ctx;
  Module m("", ctx);

#define MODULE_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7) \
  EXPECT_FALSE(has##NAME##Attr(m));                                            \
                                                                               \
  add##NAME##Attr(m, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3),   \
                  get<ETY4>(4), get<ETY5>(5), get<ETY6>(6), get<ETY7>(7));     \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(m), get<ETY0>(0));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(m), get<ETY1>(1));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(m), get<ETY2>(2));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(m), get<ETY3>(3));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(m), get<ETY4>(4));                   \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(m), get<ETY5>(5));                   \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(m), get<ETY6>(6));                   \
  EXPECT_EQ(get##ENAME7##From##NAME##Attr(m), get<ETY7>(7));                   \
                                                                               \
  add##NAME##Attr(m, get<ETY0>(7), get<ETY1>(6), get<ETY2>(5), get<ETY3>(4),   \
                  get<ETY4>(3), get<ETY5>(2), get<ETY6>(1), get<ETY7>(0));     \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(m), get<ETY0>(7));                   \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(m), get<ETY1>(6));                   \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(m), get<ETY2>(5));                   \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(m), get<ETY3>(4));                   \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(m), get<ETY4>(3));                   \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(m), get<ETY5>(2));                   \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(m), get<ETY6>(1));                   \
  EXPECT_EQ(get##ENAME7##From##NAME##Attr(m), get<ETY7>(0));                   \
                                                                               \
  remove##NAME##Attr(m);                                                       \
  EXPECT_FALSE(has##NAME##Attr(m));

#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

} // namespace
