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

// In some cases, the value of an attribute must be in a range. We cannot use
// a purely random value to test since it will likely fail. This should be
// specialized for specific attribute kinds.
template <typename T, ModuleAttrKind Attr> static T get(unsigned idx) {
  return get_<T>(idx);
}

// In some cases, it is difficult to construct a valid attribute - for instance
// if the attribute initializer must be valid bitcode. In such cases, we test
// everything but the verifier. lit tests must be added to ensure that the
// verification works correctly.
[[maybe_unused]]
static constexpr bool verifyAttr(ModuleAttrKind attr) {
  return true;
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

#define DECLS(OBJ)                                                             \
  LLVMContext ctx;                                                             \
  Module OBJ("", ctx);

TEST(KitModuleAttrs, verifyGeneric) {
  DECLS(m);
#define MODULE_ATTR_0(NAME, IRNAME, ...)                                       \
  TEST_GENERIC_VERIFY_0(m, ModuleAttrKind, NAME, IRNAME)
#define MODULE_ATTR(NAME, IRNAME, ...)                                         \
  TEST_GENERIC_VERIFY_N(m, ModuleAttrKind, NAME, IRNAME)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attrsGeneric) {
  DECLS(m);

#define MODULE_ATTR_0(NAME, ...) TEST_GENERIC_ATTR_0(m, ModuleAttrKind, NAME)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"

#define MODULE_ATTR_0(...)
#define MODULE_ATTR(NAME, ...) TEST_GENERIC_ATTR_N(m, ModuleAttrKind, NAME)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr0) {
  DECLS(m);
#define MODULE_ATTR_0(...) TEST_ATTR_0(m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr1) {
  DECLS(m);
#define MODULE_ATTR_1(...) TEST_ATTR_1(m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr2) {
  DECLS(m);
#define MODULE_ATTR_2(...) TEST_ATTR_2(m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr3) {
  DECLS(m);
#define MODULE_ATTR_3(...) TEST_ATTR_3(m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr4) {
  DECLS(m);
#define MODULE_ATTR_4(...) TEST_ATTR_4(m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr5) {
  DECLS(m);
#define MODULE_ATTR_5(...) TEST_ATTR_5(m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr6) {
  DECLS(m);
#define MODULE_ATTR_6(...) TEST_ATTR_6(m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr7) {
  DECLS(m);
#define MODULE_ATTR_7(...) TEST_ATTR_7(m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attr8) {
  DECLS(m);
#define MODULE_ATTR_8(...) TEST_ATTR_8(m, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attrLoop) {
  DECLS_LOOP(m, loopF, loopG, lis);
#define MODULE_ATTR_LOOP(...)                                                  \
  TEST_ATTR_LOOP(m, loopF, loopG, lis, ModuleAttrKind, __VA_ARGS__)
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attrRange) {
  DECLS(m);
  TEST_ATTR_ATTRS(m)
}

} // namespace
