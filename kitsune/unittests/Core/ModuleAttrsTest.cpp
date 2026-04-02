//===- ModuleAttrsTest.cpp - Unit tests for Kitsune's module attributes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ModuleAttrs.h"
#include "Core/AttrsImpl.h"
#include "Core/ModuleAttrsImpl.h"
#include "Core/VerifierImpl.h"
#include "TestAttrsCommon.h"
#include "kitsune/Core/ModuleUtils.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Override get<>() and verifyAttr<> if needed. See documentation in the base
// class for details.
class KitModuleAttrs : public TestAttrsBase<KitModuleAttrs, ModuleAttrKind> {};

TEST_F(KitModuleAttrs, attrName) {
#define MODULE_ATTR(NAME, IRNAME, ...)                                         \
  EXPECT_EQ(getAttrName(ModuleAttrKind::NAME), IRNAME);                        \
  EXPECT_TRUE(getAttrName(ModuleAttrKind::NAME).starts_with("kit.module."));
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST_F(KitModuleAttrs, attrKind) {
  EXPECT_EQ(getModuleAttrKind("balliol"), std::nullopt);
#define MODULE_ATTR(NAME, IRNAME, ...)                                         \
  EXPECT_EQ(getModuleAttrKind(IRNAME), ModuleAttrKind::NAME);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST_F(KitModuleAttrs, verifyGeneric) {
  DECLS;
#define MODULE_ATTR_0(NAME, IRNAME, ...)                                       \
  TEST_GENERIC_VERIFY_0(*m, ModuleAttrKind, NAME, IRNAME)
#define MODULE_ATTR(NAME, IRNAME, ...)                                         \
  TEST_GENERIC_VERIFY_N(*m, ModuleAttrKind, NAME, IRNAME)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST_F(KitModuleAttrs, attrsGeneric) {
  DECLS;

#define MODULE_ATTR_0(NAME, ...) TEST_GENERIC_ATTR_0(*m, ModuleAttrKind, NAME)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_0(...)
#define MODULE_ATTR(NAME, ...) TEST_GENERIC_ATTR_N(*m, ModuleAttrKind, NAME)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST_F(KitModuleAttrs, attr0) {
  DECLS;
#define MODULE_ATTR_0(...) TEST_ATTR_0(*m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST_F(KitModuleAttrs, attr1) {
  DECLS;
#define MODULE_ATTR_1(...) TEST_ATTR_1(*m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST_F(KitModuleAttrs, attr2) {
  DECLS;
#define MODULE_ATTR_2(...) TEST_ATTR_2(*m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST_F(KitModuleAttrs, attr3) {
  DECLS;
#define MODULE_ATTR_3(...) TEST_ATTR_3(*m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST_F(KitModuleAttrs, attr4) {
  DECLS;
#define MODULE_ATTR_4(...) TEST_ATTR_4(*m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST_F(KitModuleAttrs, attr5) {
  DECLS;
#define MODULE_ATTR_5(...) TEST_ATTR_5(*m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST_F(KitModuleAttrs, attr6) {
  DECLS;
#define MODULE_ATTR_6(...) TEST_ATTR_6(*m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST_F(KitModuleAttrs, attr7) {
  DECLS;
#define MODULE_ATTR_7(...) TEST_ATTR_7(*m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST_F(KitModuleAttrs, attr8) {
  DECLS;
#define MODULE_ATTR_8(...) TEST_ATTR_8(*m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST_F(KitModuleAttrs, attrRange) {
  DECLS;
  TEST_ATTR_ATTRS(*m)
}

} // namespace
