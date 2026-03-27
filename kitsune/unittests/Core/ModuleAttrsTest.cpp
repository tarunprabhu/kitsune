//===- ModuleAttrsTest.cpp - Unit tests for Kitsune's module attributes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ModuleAttrs.h"
#include "TestAttrsCommon.h"
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/ModuleUtils.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

static void addMetadata(Module &m, StringRef attrName,
                        ArrayRef<Metadata *> attrVals) {
  LLVMContext &ctx = m.getContext();
  MDNode *attrList = getAttrList(m);
  MDNode *newAttrList = getAttrListWith(attrName, attrVals, attrList, ctx);

  NamedMDNode *nmd = m.getOrInsertNamedMetadata("kit.module");
  if (nmd->getNumOperands())
    nmd->setOperand(0, newAttrList);
  else
    nmd->addOperand(newAttrList);
}

// Create metadata consisting of `n` "empty" operands.
static void addMetadata(Module &m, StringRef attrName, unsigned n) {
  LLVMContext &ctx = m.getContext();
  MDNode *mdEmpty = MDNode::get(ctx, {});
  SmallVector<Metadata *, 8> attrVals;

  attrVals.append(n, mdEmpty);
  addMetadata(m, attrName, attrVals);
}

TEST(KitModuleAttrs, attrName) {
#define MODULE_ATTR(NAME, IRNAME, ...)                                         \
  EXPECT_EQ(getAttrName(ModuleAttrKind::NAME), IRNAME);                        \
  EXPECT_TRUE(getAttrName(ModuleAttrKind::NAME).starts_with("kit.module."));
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attrKind) {
  EXPECT_EQ(getModuleAttrKind("balliol"), std::nullopt);
#define MODULE_ATTR(NAME, IRNAME, ...)                                         \
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
#define MODULE_ATTR_0(NAME, IRNAME, ...)                                       \
  TEST_GENERIC_VERIFY_0(os, m, ModuleAttrKind, NAME, IRNAME)
#define MODULE_ATTR(NAME, IRNAME, ...)                                         \
  TEST_GENERIC_VERIFY_N(os, m, ModuleAttrKind, NAME, IRNAME)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attrsGeneric) {
  DECLS(os, m);

#define MODULE_ATTR_0(NAME, ...) TEST_GENERIC_ATTR_0(m, ModuleAttrKind, NAME)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_0(...)
#define MODULE_ATTR(NAME, ...) TEST_GENERIC_ATTR_N(m, ModuleAttrKind, NAME)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr0) {
  DECLS(os, m);
#define MODULE_ATTR_0(...) TEST_ATTR_0(os, m, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr1) {
  DECLS(os, m);
#define MODULE_ATTR_1(...) TEST_ATTR_1(os, m, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr2) {
  DECLS(os, m);
#define MODULE_ATTR_2(...) TEST_ATTR_2(os, m, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr3) {
  DECLS(os, m);
#define MODULE_ATTR_3(...) TEST_ATTR_3(os, m, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr4) {
  DECLS(os, m);
#define MODULE_ATTR_4(...) TEST_ATTR_4(os, m, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr5) {
  DECLS(os, m);
#define MODULE_ATTR_5(...) TEST_ATTR_5(os, m, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr6) {
  DECLS(os, m);
#define MODULE_ATTR_6(...) TEST_ATTR_6(os, m, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr7) {
  DECLS(os, m);
#define MODULE_ATTR_7(...) TEST_ATTR_7(os, m, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr8) {
  DECLS(os, m);
#define MODULE_ATTR_8(...) TEST_ATTR_8(os, m, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attrLoop) {
  DECLS_LOOP(os, m, loopF, loopG, lis)
#define MODULE_ATTR_LOOP(...)                                                  \
  TEST_ATTR_LOOP(os, m, loopF, loopG, lis, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

} // namespace
