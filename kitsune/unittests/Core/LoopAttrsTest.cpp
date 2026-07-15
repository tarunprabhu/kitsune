//===- LoopAttrsTest.cpp - Unit tests for Kitsune's loop attributes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/LoopAttrs.h"
#include "Core/AttrsImpl.h"
#include "Core/LoopAttrsImpl.h"
#include "Core/VerifierImpl.h"
#include "TestAttrsCommon.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Override get<>() and verifyAttr<> if needed. See documentation in the base
// class for details.
class KitLoopAttrs : public TestAttrsBase<KitLoopAttrs, LoopAttrKind> {
public:
  // This is required in order to specialize for LoopAttrKind::ThreadsPerBlock
  // below.
  template <typename T, LoopAttrKind Attr> T get(unsigned idx) {
    if constexpr (std::is_same_v<T, uint32_t>) {
      static constexpr int32_t pool[] = {0, 4, 16, 64, 128, 256, 512, 1024};
      return pool[idx % (sizeof(pool) / sizeof(int32_t))];
    } else {
      return TestAttrsBase::get<T>(idx);
    }
  }
};

TEST_F(KitLoopAttrs, attrName) {
#define LOOP_ATTR(NAME, IRNAME, ...)                                           \
  EXPECT_EQ(getAttrName(LoopAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(LoopAttrKind::NAME).starts_with("tapir.loop."));
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST_F(KitLoopAttrs, attrKind) {
  EXPECT_EQ(getLoopAttrKind("wolfson"), std::nullopt);
#define LOOP_ATTR(NAME, IRNAME, ...)                                           \
  EXPECT_EQ(getLoopAttrKind(IRNAME), LoopAttrKind::NAME);
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST_F(KitLoopAttrs, verifyGeneric) {
  DECLS;
#define LOOP_ATTR_0(NAME, IRNAME, ...)                                         \
  TEST_GENERIC_VERIFY_0(*loop, LoopAttrKind, NAME, IRNAME)
#define LOOP_ATTR(NAME, IRNAME, ...)                                           \
  TEST_GENERIC_VERIFY_N(*loop, LoopAttrKind, NAME, IRNAME)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST_F(KitLoopAttrs, attrsGeneric) {
  DECLS;

#define LOOP_ATTR_0(NAME, ...) TEST_GENERIC_ATTR_0(*loop, LoopAttrKind, NAME)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"

#define LOOP_ATTR_0(...)
#define LOOP_ATTR(NAME, ...) TEST_GENERIC_ATTR_N(*loop, LoopAttrKind, NAME)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST_F(KitLoopAttrs, attr0) {
  DECLS;
#define LOOP_ATTR_0(...) TEST_ATTR_0(*loop, LoopAttrKind, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST_F(KitLoopAttrs, attr1) {
  DECLS;
#define LOOP_ATTR_1(...) TEST_ATTR_1(*loop, LoopAttrKind, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST_F(KitLoopAttrs, attr2) {
  DECLS;
#define LOOP_ATTR_2(...) TEST_ATTR_2(*loop, LoopAttrKind, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST_F(KitLoopAttrs, attr3) {
  DECLS;
#define LOOP_ATTR_3(...) TEST_ATTR_3(*loop, LoopAttrKind, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST_F(KitLoopAttrs, attr4) {
  DECLS;
#define LOOP_ATTR_4(...) TEST_ATTR_4(*loop, LoopAttrKind, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST_F(KitLoopAttrs, attr5) {
  DECLS;
#define LOOP_ATTR_5(...) TEST_ATTR_5(*loop, LoopAttrKind, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST_F(KitLoopAttrs, attr6) {
  DECLS;
#define LOOP_ATTR_6(...) TEST_ATTR_6(*loop, LoopAttrKind, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST_F(KitLoopAttrs, attr7) {
  DECLS;
#define LOOP_ATTR_7(...) TEST_ATTR_7(*loop, LoopAttrKind, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST_F(KitLoopAttrs, attr8) {
  DECLS;
#define LOOP_ATTR_8(...) TEST_ATTR_8(*loop, LoopAttrKind, __VA_ARGS__)
#define GET_LOOP_ATTRS
#include "kitsune/Core/LoopAttrs.inc"
}

TEST_F(KitLoopAttrs, attrRange) {
  DECLS;
  TEST_ATTR_ATTRS(*loop)
}

} // namespace
