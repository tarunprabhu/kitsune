//===- FuncAttrsTest.cpp - Unit tests for Kitsune's function attributes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/FuncAttrs.h"
#include "Core/AttrsImpl.h"
#include "Core/FuncAttrsImpl.h"
#include "Core/VerifierImpl.h"
#include "TestAttrsCommon.h"
#include "kitsune/Core/FuncUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Override get<>() and verifyAttr<> if needed. See documentation in the base
// class for details.
class KitFuncAttrs : public TestAttrsBase<KitFuncAttrs, FuncAttrKind> {
public:
  template <typename T, FuncAttrKind Attr> T get(unsigned idx) {
    if constexpr (Attr == FuncAttrKind::Kernel) {
      static constexpr T pool[] = {1, 2, 3};
      return pool[idx % (sizeof(pool) / sizeof(T))];
    } else {
      return TestAttrsBase::get<T>(idx);
    }
  }
};

TEST_F(KitFuncAttrs, attrName) {
#define FUNC_ATTR(NAME, IRNAME, ...)                                           \
  EXPECT_EQ(getAttrName(FuncAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(FuncAttrKind::NAME).starts_with("kit.func."));
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, attrKind) {
  EXPECT_EQ(getFuncAttrKind("keble"), std::nullopt);
#define FUNC_ATTR(NAME, IRNAME, ...)                                           \
  EXPECT_EQ(getFuncAttrKind(IRNAME), FuncAttrKind::NAME);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, verifyGeneric) {
  DECLS;
#define FUNC_ATTR_0(NAME, IRNAME, ...)                                         \
  TEST_GENERIC_VERIFY_0(*f, FuncAttrKind, NAME, IRNAME)
#define FUNC_ATTR(NAME, IRNAME, ...)                                           \
  TEST_GENERIC_VERIFY_N(*f, FuncAttrKind, NAME, IRNAME)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, attrsGeneric) {
  DECLS;

#define FUNC_ATTR_0(NAME, ...) TEST_GENERIC_ATTR_0(*f, FuncAttrKind, NAME)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_0(...)
#define FUNC_ATTR(NAME, ...) TEST_GENERIC_ATTR_N(*f, FuncAttrKind, NAME)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, attrList) {
  DECLS;
#define FUNC_ATTR_L(...) TEST_ATTR_L(*f, FuncAttrKind, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, attrSet) {
  DECLS;
#define FUNC_ATTR_S(...) TEST_ATTR_S(*f, FuncAttrKind, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, attr0) {
  DECLS;
#define FUNC_ATTR_0(...) TEST_ATTR_0(*f, FuncAttrKind, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, attr1) {
  DECLS;
#define FUNC_ATTR_1(...) TEST_ATTR_1(*f, FuncAttrKind, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, attr2) {
  DECLS;
#define FUNC_ATTR_2(...) TEST_ATTR_2(*f, FuncAttrKind, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, attr3) {
  DECLS;
#define FUNC_ATTR_3(...) TEST_ATTR_3(*f, FuncAttrKind, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, attr4) {
  DECLS;
#define FUNC_ATTR_4(...) TEST_ATTR_4(*f, FuncAttrKind, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, attr5) {
  DECLS;
#define FUNC_ATTR_5(...) TEST_ATTR_5(*f, FuncAttrKind, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, attr6) {
  DECLS;
#define FUNC_ATTR_6(...) TEST_ATTR_6(*f, FuncAttrKind, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, attr7) {
  DECLS;
#define FUNC_ATTR_7(...) TEST_ATTR_7(*f, FuncAttrKind, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, attr8) {
  DECLS;
#define FUNC_ATTR_8(...) TEST_ATTR_8(*f, FuncAttrKind, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST_F(KitFuncAttrs, attrRange) {
  DECLS;
  TEST_ATTR_ATTRS(*f)
}

} // namespace
