//==- InstAttrsTest.cpp - Unit tests for Kitsune's instruction attributes --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/InstAttrs.h"
#include "Core/AttrsImpl.h"
#include "Core/InstAttrsImpl.h"
#include "Core/VerifierImpl.h"
#include "TestAttrsCommon.h"
#include "kitsune/Core/InstUtils.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Override get<>() and verifyAttr<> if needed. See documentation in the base
// class for details.
class KitInstAttrs : public TestAttrsBase<KitInstAttrs, InstAttrKind> {};

TEST_F(KitInstAttrs, attrName) {
#define INST_ATTR(NAME, IRNAME, ...)                                           \
  EXPECT_EQ(getAttrName(InstAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(InstAttrKind::NAME).starts_with("kit.inst."));
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, attrKind) {
  EXPECT_EQ(getInstAttrKind("queen's"), std::nullopt);
#define INST_ATTR(NAME, IRNAME, ...)                                           \
  EXPECT_EQ(getInstAttrKind(IRNAME), InstAttrKind::NAME);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, verifyGeneric) {
  DECLS;
#define INST_ATTR_0(NAME, IRNAME, ...)                                         \
  TEST_GENERIC_VERIFY_0(*inst, InstAttrKind, NAME, IRNAME)
#define INST_ATTR(NAME, IRNAME, ...)                                           \
  TEST_GENERIC_VERIFY_N(*inst, InstAttrKind, NAME, IRNAME)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, attrsGeneric) {
  DECLS;

#define INST_ATTR_0(NAME, ...) TEST_GENERIC_ATTR_0(*inst, InstAttrKind, NAME)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_0(...)
#define INST_ATTR(NAME, ...) TEST_GENERIC_ATTR_N(*inst, InstAttrKind, NAME)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, attrList) {
  DECLS;
#define INST_ATTR_L(...) TEST_ATTR_L(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, attrSet) {
  DECLS;
#define INST_ATTR_S(...) TEST_ATTR_S(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, attr0) {
  DECLS;
#define INST_ATTR_0(...) TEST_ATTR_0(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, attr1) {
  DECLS;
#define INST_ATTR_1(...) TEST_ATTR_1(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, attr2) {
  DECLS;
#define INST_ATTR_2(...) TEST_ATTR_2(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, attr3) {
  DECLS;
#define INST_ATTR_3(...) TEST_ATTR_3(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, attr4) {
  DECLS;
#define INST_ATTR_4(...) TEST_ATTR_4(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, attr5) {
  DECLS;
#define INST_ATTR_5(...) TEST_ATTR_5(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, attr6) {
  DECLS;
#define INST_ATTR_6(...) TEST_ATTR_6(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, attr7) {
  DECLS;
#define INST_ATTR_7(...) TEST_ATTR_7(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, attr8) {
  DECLS;
#define INST_ATTR_8(...) TEST_ATTR_8(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST_F(KitInstAttrs, attrRange) {
  DECLS;
  TEST_ATTR_ATTRS(*inst)
}

} // namespace
