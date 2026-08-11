//===- RuntimeTraitsTest.cpp - Tests for runtime context traits -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for traits to be used with the runtime context objects. We test these
// here because they are not part of the standard traits header and are intended
// to be private to runtimes.h
//
//===----------------------------------------------------------------------===//

#include "runtimes.h"

#include "gtest/gtest.h"

using namespace kitrt;

namespace {

TEST(KitrtRuntimeTraits, hasMemberInitialize) {
  struct Static {
    static void initialize();
  };

  struct MemberRet {
    int initialize();
  };

  struct MemberVar {
    void *initialize;
  };

  struct MemberArgs {
    void initialize(int, void *);
  };

  struct MemberArg {
    void initialize(int);
  };

  struct Member {
    void initialize();
  };

  EXPECT_FALSE(detail::has_initialize<Static>::value);
  EXPECT_FALSE(detail::has_initialize<MemberRet>::value);
  EXPECT_FALSE(detail::has_initialize<MemberVar>::value);

  EXPECT_TRUE(detail::has_initialize<MemberArgs>::value);
  EXPECT_TRUE(detail::has_initialize<MemberArg>::value);
  EXPECT_TRUE(detail::has_initialize<Member>::value);
}

TEST(KitrtRuntimeTraits, hasMemberFinalize) {
  struct Static {
    static void finalize();
  };

  struct MemberArg {
    void finalize(int);
  };

  struct MemberRet {
    int finalize();
  };

  struct MemberVar {
    void *finalize;
  };

  struct Member {
    void finalize();
  };

  EXPECT_FALSE(detail::has_finalize<Static>::value);
  EXPECT_FALSE(detail::has_finalize<MemberArg>::value);
  EXPECT_FALSE(detail::has_finalize<MemberRet>::value);
  EXPECT_FALSE(detail::has_finalize<MemberVar>::value);

  EXPECT_TRUE(detail::has_finalize<Member>::value);
}

// TEST(KitrtRuntimeTraits, hasMemberGetThreadID) {
//   struct Static : public ContextBase<Static> {
//     static KitThreadID getThreadID();
//   };

//   struct MemberArg : public ContextBase<MemberArg> {
//     KitThreadID getThreadID(int);
//   };

//   struct MemberRet : public ContextBase<MemberRet> {
//     void getThreadID();
//   };

//   struct MemberVar : public ContextBase<MemberVar> {
//     KitThreadID getThreadID;
//   };

//   struct Member : public ContextBase<Member> {
//     KitThreadID getThreadID();
//   };

//   EXPECT_FALSE(detail::has_getThreadID<Static>::value);
//   EXPECT_FALSE(detail::has_getThreadID<MemberArg>::value);
//   EXPECT_FALSE(detail::has_getThreadID<MemberRet>::value);
//   EXPECT_FALSE(detail::has_getThreadID<MemberVar>::value);

//   EXPECT_TRUE(detail::has_getThreadID<Member>::value);
// }

TEST(KitrtRuntimeTraits, isContext) {
  struct BadEmpty {};

  struct BadOnlyInit {
    void initialize(void);
  };

  struct BadOnlyFini {
    void finalize(void);
  };

  struct BadFiniSig {
    void initialize(void);
    void finalize(void*);
  };

  struct Good {
    void initialize(void *);
    void finalize();
  };

  EXPECT_FALSE(detail::is_context_v<BadEmpty>);
  EXPECT_FALSE(detail::is_context_v<BadOnlyInit>);
  EXPECT_FALSE(detail::is_context_v<BadOnlyFini>);
  EXPECT_FALSE(detail::is_context_v<BadFiniSig>);

  EXPECT_TRUE(detail::is_context_v<Good>);
}

} // namespace
