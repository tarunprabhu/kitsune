//=- EmbDeviceCodeTest.cpp - Unit tests for the embedded device code object -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/EmbDeviceCode.h"
#include "CompressedBinary.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

static EmbDeviceCode make(EmbDeviceCode::Id id, const Binary &bin) {
  MemoryBufferRef memBuf = bin.getMemoryBufferRef();
  Expected<EmbDeviceCode> devCode = EmbDeviceCode::create(id, memBuf);

  EXPECT_TRUE(bool(devCode));
  return *devCode;
}

static EmbDeviceCode make(EmbDeviceCode::Id id, const Binary &bin,
                          StringRef file) {
  MemoryBufferRef memBuf(bin.getData(), file);
  Expected<EmbDeviceCode> devCode = EmbDeviceCode::create(id, memBuf);

  EXPECT_TRUE(bool(devCode));
  return *devCode;
}

TEST(EmbDeviceCodeTest, sentinels) {
  auto u = [](EmbDeviceCode::Id id) -> uint64_t { return uint64_t(id); };

  EXPECT_EQ(u(EmbDeviceCode::INVALID), 0U);

  EXPECT_EQ(u(EmbDeviceCode::AMDGPU_lo), u(EmbDeviceCode::GFX600) - 1);
  EXPECT_EQ(u(EmbDeviceCode::AMDGPU_hi), u(EmbDeviceCode::GFX1201) + 1);

  EXPECT_EQ(u(EmbDeviceCode::NVSASS_lo), u(EmbDeviceCode::SM_30) - 1);
  EXPECT_EQ(u(EmbDeviceCode::NVSASS_hi), u(EmbDeviceCode::SM_120A) + 1);

  EXPECT_EQ(u(EmbDeviceCode::NVPTX_lo), u(EmbDeviceCode::COMPUTE_30) - 1);
  EXPECT_EQ(u(EmbDeviceCode::NVPTX_hi), u(EmbDeviceCode::COMPUTE_120A) + 1);
}

TEST(EmbDeviceCodeTest, createFromInt) {
  auto pass = [](uint64_t n, EmbDeviceCode::Id expected) -> void {
    MemoryBufferRef memBuf = elfEmpty->getMemoryBufferRef();
    Expected<EmbDeviceCode> devCode = EmbDeviceCode::create(n, memBuf);

    EXPECT_TRUE(bool(devCode));
    EXPECT_EQ(devCode->getId(), expected);
  };

  auto fail = [](uint64_t n) -> void {
    MemoryBufferRef memBuf = elfEmpty->getMemoryBufferRef();
    Expected<EmbDeviceCode> devCode = EmbDeviceCode::create(n, memBuf);

    EXPECT_FALSE(bool(devCode));
  };

  pass(EmbDeviceCode::GFX600, EmbDeviceCode::GFX600);
  pass(EmbDeviceCode::GFX1201, EmbDeviceCode::GFX1201);
  pass(EmbDeviceCode::SM_30, EmbDeviceCode::SM_30);
  pass(EmbDeviceCode::SM_120A, EmbDeviceCode::SM_120A);
  pass(EmbDeviceCode::COMPUTE_30, EmbDeviceCode::COMPUTE_30);
  pass(EmbDeviceCode::COMPUTE_120A, EmbDeviceCode::COMPUTE_120A);

  fail(EmbDeviceCode::INVALID);
  fail(EmbDeviceCode::AMDGPU_lo);
  fail(EmbDeviceCode::AMDGPU_hi);
  fail(EmbDeviceCode::NVSASS_lo);
  fail(EmbDeviceCode::NVSASS_hi);
  fail(EmbDeviceCode::NVPTX_lo);
  fail(EmbDeviceCode::NVPTX_hi);
}

TEST(EmbDeviceCodeTest, createFromId) {
  auto pass = [](EmbDeviceCode::Id id) -> void {
    MemoryBufferRef memBuf = elfEmpty->getMemoryBufferRef();
    Expected<EmbDeviceCode> devCode = EmbDeviceCode::create(id, memBuf);

    EXPECT_TRUE(bool(devCode));
    EXPECT_EQ(devCode->getId(), id);
  };

  auto fail = [](EmbDeviceCode::Id id) -> void {
    MemoryBufferRef memBuf = elfEmpty->getMemoryBufferRef();
    Expected<EmbDeviceCode> devCode = EmbDeviceCode::create(id, memBuf);

    EXPECT_FALSE(bool(devCode));
  };

  pass(EmbDeviceCode::GFX600);
  pass(EmbDeviceCode::GFX1201);
  pass(EmbDeviceCode::SM_30);
  pass(EmbDeviceCode::SM_120A);
  pass(EmbDeviceCode::COMPUTE_30);
  pass(EmbDeviceCode::COMPUTE_120A);

  fail(EmbDeviceCode::INVALID);
  fail(EmbDeviceCode::AMDGPU_lo);
  fail(EmbDeviceCode::AMDGPU_hi);
  fail(EmbDeviceCode::NVSASS_lo);
  fail(EmbDeviceCode::NVSASS_hi);
  fail(EmbDeviceCode::NVPTX_lo);
  fail(EmbDeviceCode::NVPTX_hi);
}

TEST(EmbDeviceCodeTest, isArchive) {
  auto check = [](const Binary &bin, bool expected) -> void {
    EmbDeviceCode devCode = make(EmbDeviceCode::GFX90A, bin);

    EXPECT_EQ(devCode.isArchive(), expected);
  };

  check(*arEmpty, true);
  check(*elfEmpty, false);
  check(*elfShared, false);
  check(*machOObject, false);
  check(*machOShared, false);
}

TEST(EmbDeviceCodeTest, isObject) {
  auto check = [](const Binary &bin, bool expected) -> void {
    EmbDeviceCode devCode = make(EmbDeviceCode::SM_80, bin);

    EXPECT_EQ(devCode.isObject(), expected);
  };

  check(*arEmpty, false);
  check(*elfEmpty, true);
  check(*elfShared, false);
  check(*machOObject, true);
  check(*machOShared, false);
}

TEST(EmbDeviceCodeTest, isShared) {
  auto check = [](const Binary &bin, bool expected) -> void {
    EmbDeviceCode devCode = make(EmbDeviceCode::GFX1103, bin);

    EXPECT_EQ(devCode.isShared(), expected);
  };

  check(*arEmpty, false);
  check(*elfEmpty, false);
  check(*elfShared, true);
  check(*machOObject, false);
  check(*machOShared, true);
}

TEST(EmbDeviceCodeTest, getName) {
  auto check = [](EmbDeviceCode::Id id, const Binary &bin, StringRef file,
                  StringRef exp) -> void {
    EmbDeviceCode devCode = make(id, bin, file);

    EXPECT_EQ(devCode.getName(), exp);
  };

  check(EmbDeviceCode::GFX600, *elfEmpty, "f1.o", "f1-gfx600.o");
  check(EmbDeviceCode::GFX90C, *arEmpty, "f2.o", "f2-gfx90c.a");
  check(EmbDeviceCode::SM_30, *arEmpty, "f3.so", "f3-sm_30.a");
  check(EmbDeviceCode::SM_120A, *elfEmpty, "f4.o", "f4-sm_120a.cubin");
  check(EmbDeviceCode::COMPUTE_30, *arEmpty, "f5.o", "f5-compute_30.a");
  check(EmbDeviceCode::COMPUTE_120A, *elfEmpty, "f6.a", "f6-compute_120a.ptx");
}

TEST(EmbDeviceCodeTest, getArch) {
  auto check = [](EmbDeviceCode::Id id) -> void {
    EmbDeviceCode devCode = make(id, *elfEmpty);
    StringRef arch = devCode.getArch();
    Expected<EmbDeviceCode::Id> id2 = EmbDeviceCode::getIdFor(arch);

    EXPECT_TRUE(bool(id2));
    EXPECT_EQ(*id2, id);
  };

  // This checks the first and last device id for each "class" of device. It
  // would be good to keep this up to date with the list of known device id's
  // in EmbDeviceCode::Id, but it is not strictly necessary.
  check(EmbDeviceCode::GFX600);
  check(EmbDeviceCode::GFX1201);
  check(EmbDeviceCode::SM_30);
  check(EmbDeviceCode::SM_120A);
  check(EmbDeviceCode::COMPUTE_30);
  check(EmbDeviceCode::COMPUTE_120A);
}

TEST(EmbDeviceCodeTest, getFormat) {
  auto check = [](EmbDeviceCode::Id id,
                  EmbDeviceCode::BinaryFormat expected) -> void {
    EXPECT_EQ(make(id, *elfEmpty).getBinaryFormat(), expected);
  };

  // This checks the first and last device id for each "class" of device. It
  // would be good to keep this up to date with the list of known device id's
  // in EmbDeviceCode::Id, but it is not strictly necessary.
  check(EmbDeviceCode::GFX600, EmbDeviceCode::AMDGPU);
  check(EmbDeviceCode::GFX1201, EmbDeviceCode::AMDGPU);
  check(EmbDeviceCode::SM_30, EmbDeviceCode::NVSASS);
  check(EmbDeviceCode::SM_120, EmbDeviceCode::NVSASS);
  check(EmbDeviceCode::COMPUTE_30, EmbDeviceCode::NVPTX);
  check(EmbDeviceCode::COMPUTE_120A, EmbDeviceCode::NVPTX);
}

TEST(EmbDeviceCodeTest, getTTID) {
  auto check = [](EmbDeviceCode::Id id, const Binary &bin,
                  TTID expected) -> void {
    EXPECT_EQ(make(id, bin).getTTID(), expected);
  };

  check(EmbDeviceCode::SM_72, *elfCuda, TTID::Cuda);
  check(EmbDeviceCode::GFX906, *elfHip, TTID::Hip);
}

TEST(EmbDeviceCodeTest, getIdForString) {
  auto check_eq = [](StringRef s, EmbDeviceCode::Id expected) -> void {
    Expected<EmbDeviceCode::Id> id = EmbDeviceCode::getIdFor(s);

    EXPECT_TRUE(bool(id));
    EXPECT_EQ(*id, expected);
  };

  // This checks the first and last device id for each "class" of device. It
  // would be good to keep this up to date with the list of known device id's
  // in EmbDeviceCode::Id, but it is not strictly necessary.
  check_eq("gfx600", EmbDeviceCode::GFX600);
  check_eq("gfx1201", EmbDeviceCode::GFX1201);
  check_eq("sm_30", EmbDeviceCode::SM_30);
  check_eq("sm_120a", EmbDeviceCode::SM_120A);
  check_eq("compute_30", EmbDeviceCode::COMPUTE_30);
  check_eq("compute_120a", EmbDeviceCode::COMPUTE_120A);

  // The device code id's are case sensitive.
  EXPECT_FALSE(EmbDeviceCode::getIdFor("GFX600"));
  EXPECT_FALSE(EmbDeviceCode::getIdFor("Gfx1201"));
  EXPECT_FALSE(EmbDeviceCode::getIdFor("SM_30"));
  EXPECT_FALSE(EmbDeviceCode::getIdFor("Sm_120A"));
  EXPECT_FALSE(EmbDeviceCode::getIdFor("COMPUTE_30"));
  EXPECT_FALSE(EmbDeviceCode::getIdFor("Compute_30"));

  // Unknown strings
  EXPECT_FALSE(EmbDeviceCode::getIdFor(""));
  EXPECT_FALSE(EmbDeviceCode::getIdFor("gfx500"));
  EXPECT_FALSE(EmbDeviceCode::getIdFor("sm_20"));
  EXPECT_FALSE(EmbDeviceCode::getIdFor("compute_120b"));
}
