//===- ObjectUtilsTest.cpp - Tests for LLVM's binary object utilities -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/ObjectUtils.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

static const StringRef bArchive("!<arch>\n", 8);

// The minimum size of ELF objects at this time is 18.
static const StringRef bELFExecutable("\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\2\0",
                                      18);
static const StringRef bELFObject("\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\1\0", 18);
static const StringRef bELFShared("\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\3\0", 18);

// The minimum size of MachO objects at this time is 32.
static const StringRef bMachOExecutable(
    "\317\372\355\376\7\0\0\1\3\0\0\0\2\0\0\0\14\0\0\0\340\2\0\0\0\40\0\0\0\0",
    32);
static const StringRef bMachOObject("\317\372\355\376\7\0\0\1\3\0\0\0\1\0\0\0\2"
                                    "\0\0\0\250\0\0\205\0\2\0\0\0\0\0",
                                    32);
static const StringRef bMachOShared("\317\372\355\376\7\0\0\1\3\0\0\0\6\0\0\0"
                                    "\12\0\0\0\240\1\0\0\205\0\20\0\0\0",
                                    32);

static const MemoryBufferRef mArchive(bArchive, "archive buffer");
static const MemoryBufferRef mELFExecutable(bELFExecutable, "ELF executable");
static const MemoryBufferRef mELFObject(bELFObject, "ELF object buffer");
static const MemoryBufferRef mELFShared(bELFShared, "ELF shared buffer");
static const MemoryBufferRef mMachOExecutable(bMachOExecutable,
                                              "MachO executable");
static const MemoryBufferRef mMachOObject(bMachOObject, "MachO object buffer");
static const MemoryBufferRef mMachOShared(bMachOShared, "MachO shared buffer");

class ObjectUtilsTest : public testing::Test {
protected:
  std::unique_ptr<Archive> archive;

protected:
  ObjectUtilsTest() {
    Expected<std::unique_ptr<Archive>> archiveOrErr =
        Archive::create(MemoryBufferRef(bArchive, "archive"));
    if (not archiveOrErr)
      (void)archiveOrErr.takeError();
    this->archive = std::move(*archiveOrErr);
  }
};

TEST_F(ObjectUtilsTest, isArchive) {
  auto check = [](const MemoryBufferRef &memBuf, bool expected) -> void {
    StringRef strRef = memBuf.getBuffer();

    EXPECT_EQ(isArchive(memBuf), expected);
    EXPECT_EQ(isArchive(strRef), expected);
  };

  EXPECT_TRUE(isArchive(*archive));
  check(mArchive, true);
  check(mELFExecutable, false);
  check(mELFObject, false);
  check(mELFShared, false);
  check(mMachOExecutable, false);
  check(mMachOObject, false);
  check(mMachOShared, false);
}

TEST_F(ObjectUtilsTest, isExecutable) {
  auto check = [](const MemoryBufferRef &memBuf, bool expected) -> void {
    StringRef strRef = memBuf.getBuffer();

    EXPECT_EQ(isExecutable(memBuf), expected);
    EXPECT_EQ(isExecutable(strRef), expected);
  };

  EXPECT_FALSE(isExecutable(*archive));
  check(mArchive, false);
  check(mELFExecutable, true);
  check(mELFObject, false);
  check(mELFShared, false);
  check(mMachOExecutable, true);
  check(mMachOObject, false);
  check(mMachOShared, false);
}

TEST_F(ObjectUtilsTest, isObject) {
  auto check = [](const MemoryBufferRef &memBuf, bool expected) -> void {
    StringRef strRef = memBuf.getBuffer();

    EXPECT_EQ(isObject(memBuf), expected);
    EXPECT_EQ(isObject(strRef), expected);
  };

  EXPECT_FALSE(isObject(*archive));
  check(mArchive, false);
  check(mELFExecutable, false);
  check(mELFObject, true);
  check(mELFShared, false);
  check(mMachOExecutable, false);
  check(mMachOObject, true);
  check(mMachOShared, false);
}

TEST_F(ObjectUtilsTest, isShared) {
  auto check = [](const MemoryBufferRef &memBuf, bool expected) -> void {
    StringRef strRef = memBuf.getBuffer();

    EXPECT_EQ(isShared(memBuf), expected);
    EXPECT_EQ(isShared(strRef), expected);
  };

  EXPECT_FALSE(isShared(*archive));
  check(mArchive, false);
  check(mELFExecutable, false);
  check(mELFObject, false);
  check(mELFShared, true);
  check(mMachOExecutable, false);
  check(mMachOObject, false);
  check(mMachOShared, true);
}
