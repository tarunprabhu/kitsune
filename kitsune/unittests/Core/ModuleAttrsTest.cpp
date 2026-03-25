//===- ModuleAttrsTest.cpp - Unit tests for Kitsune's module attributes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ModuleAttrs.h"
#include "TestAttrsCommon.h"
#include "TestValues.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

// The standard accessors do not allow us to create invalid attributes. To
// create one, we have to know how these are added to the function. This is not
// unreasonable since the create functions are a fairly thin wrappers around
// LLVM's existing support.
static void addMetadata(Module &m, StringRef name, ArrayRef<Metadata *> ops) {
  if (NamedMDNode *nmd = m.getNamedMetadata(name))
    m.eraseNamedMetadata(nmd);

  LLVMContext &ctx = m.getContext();
  NamedMDNode *nmd = m.getOrInsertNamedMetadata(name);
  for (Metadata *op : ops)
    nmd->addOperand(MDNode::get(ctx, op));
}

// Create metadata consisting of `n` "empty" operands.
static void addMetadata(Module &m, StringRef name, unsigned n) {
  if (NamedMDNode *nmd = m.getNamedMetadata(name))
    m.eraseNamedMetadata(nmd);

  LLVMContext &ctx = m.getContext();
  NamedMDNode *nmd = m.getOrInsertNamedMetadata(name);
  for (unsigned i = 0; i < n; ++i)
    nmd->addOperand(MDNode::get(ctx, {}));
}

TEST(KitModuleAttrs, attrName) {
#define MODULE_ATTR(NAME, IRNAME, TYPE)                                        \
  EXPECT_EQ(getAttrName(ModuleAttrKind::NAME), IRNAME);                        \
  EXPECT_TRUE(getAttrName(ModuleAttrKind::NAME).starts_with("kit.module."));
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attrKind) {
  EXPECT_EQ(getModuleAttrKind("balliol"), std::nullopt);
#define MODULE_ATTR(NAME, IRNAME, TYPE)                                        \
  EXPECT_EQ(getModuleAttrKind(IRNAME), ModuleAttrKind::NAME);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

#define DECLS(OS, OBJ)                                                         \
  std::string buf;                                                             \
  raw_string_ostream OS(buf);                                                  \
  LLVMContext ctx;                                                             \
  Module OBJ("", ctx);

TEST(KitModuleAttrs, verifyGeneric) {
  DECLS(os, m);
#define MODULE_ATTR_0(NAME, IRNAME)                                            \
  TEST_GENERIC_VERIFY_0(os, m, ModuleAttrKind, NAME, IRNAME);
#define MODULE_ATTR(NAME, IRNAME, TYPE)                                        \
  TEST_GENERIC_VERIFY_N(os, m, ModuleAttrKind, NAME, IRNAME);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attrsGeneric) {
  DECLS(os, m);

#define MODULE_ATTR_0(NAME, IRNAME)                                            \
  TEST_GENERIC_ATTR_0(m, ModuleAttrKind, NAME);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_0(NAME, IRNAME)
#define MODULE_ATTR(NAME, IRNAME, TYPE)                                        \
  TEST_GENERIC_ATTR_N(m, ModuleAttrKind, NAME);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr0) {
  DECLS(os, m);
#define MODULE_ATTR_0(NAME, IRNAME) TEST_ATTR_0(os, m, NAME, IRNAME);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr1) {
  DECLS(os, m);
#define MODULE_ATTR_1(NAME, IRNAME, TYPE)                                      \
  TEST_ATTR_1(os, m, NAME, IRNAME, TYPE);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr2) {
  DECLS(os, m);
#define MODULE_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)      \
  TEST_ATTR_2(os, m, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr3) {
  DECLS(os, m);
#define MODULE_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2)                                       \
  TEST_ATTR_3(os, m, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2, \
              ENAME2, EN2);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr4) {
  DECLS(os, m);
#define MODULE_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)                    \
  TEST_ATTR_4(os, m, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2, \
              ENAME2, EN2, ETY3, ENAME3, EN3);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr5) {
  DECLS(os, m);
#define MODULE_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4) \
  TEST_ATTR_5(os, m, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2, \
              ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr6) {
  DECLS(os, m);
#define MODULE_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5)                                       \
  TEST_ATTR_6(os, m, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2, \
              ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5, ENAME5, \
              EN5);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr7) {
  DECLS(os, m);
#define MODULE_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5, ETY6, ENAME6, EN6)                    \
  TEST_ATTR_7(os, m, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2, \
              ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5, ENAME5, \
              EN5, ETY6, ENAME6, EN6);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr8) {
  DECLS(os, m);
#define MODULE_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
                      ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, \
                      ETY5, ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7) \
  TEST_ATTR_8(os, m, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2, \
              ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5, ENAME5, \
              EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

} // namespace
