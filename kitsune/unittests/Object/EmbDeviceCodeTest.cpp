//=- EmbDeviceCodeTest.cpp - Unit tests for the embedded device code object -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/EmbDeviceCode.h"

#include "gtest/gtest.h"

using namespace llvm;

static constexpr StringRef elf("\177ELF\2\1\1\0\0\0\0\0\0\0\0\0\1\0>\0", 20);
static constexpr StringRef archive("!<arch>\n", 8);

TEST(EmbDeviceCodeTest, sentinels) {
  auto u = [](EmbDeviceCode::Id id) -> uint64_t { return uint64_t(id); };

  EXPECT_EQ(u(EmbDeviceCode::AMDGPU_lo), u(EmbDeviceCode::GFX600) - 1);
  EXPECT_EQ(u(EmbDeviceCode::AMDGPU_hi), u(EmbDeviceCode::GFX1201) + 1);

  EXPECT_EQ(u(EmbDeviceCode::NVSASS_lo), u(EmbDeviceCode::SM_30) - 1);
  EXPECT_EQ(u(EmbDeviceCode::NVSASS_hi), u(EmbDeviceCode::SM_120A) + 1);

  EXPECT_EQ(u(EmbDeviceCode::NVPTX_lo), u(EmbDeviceCode::COMPUTE_30) - 1);
  EXPECT_EQ(u(EmbDeviceCode::NVPTX_hi), u(EmbDeviceCode::COMPUTE_120A) + 1);
}

TEST(EmbDeviceCodeTest, getName) {
  auto check = [](EmbDeviceCode::Id id, StringRef code, StringRef file,
                  StringRef exp) -> void {
    EmbDeviceCode devCode(id, code, file);

    EXPECT_EQ(devCode.getName(), exp);
  };

  check(EmbDeviceCode::GFX600, elf, "f1.o", "f1-gfx600.o");
  check(EmbDeviceCode::GFX90C, archive, "f2.o", "f2-gfx90c.a");
  check(EmbDeviceCode::SM_30, archive, "f3.so", "f3-sm_30.a");
  check(EmbDeviceCode::SM_120A, elf, "f4.o", "f4-sm_120a.cubin");
  check(EmbDeviceCode::COMPUTE_30, archive, "f5.o", "f5-compute_30.a");
  check(EmbDeviceCode::COMPUTE_120A, elf, "f6.a", "f6-compute_120a.ptx");
}

TEST(EmbDeviceCodeTest, getArch) {
  auto check_eq = [](EmbDeviceCode::Id id) -> void {
    EmbDeviceCode devCode(id, elf, "");
    StringRef arch = devCode.getArch();
    Expected<EmbDeviceCode::Id> idOrErr = EmbDeviceCode::getIdFor(arch);

    EXPECT_TRUE(bool(idOrErr));
    EXPECT_EQ(idOrErr.get(), id);
  };

  // This checks the first and last device id for each "class" of device. It
  // would be good to keep this up to date with the list of known device id's
  // in EmbDeviceCode::Id, but it is not strictly necessary.
  check_eq(EmbDeviceCode::GFX600);
  check_eq(EmbDeviceCode::GFX1201);
  check_eq(EmbDeviceCode::SM_30);
  check_eq(EmbDeviceCode::SM_120A);
  check_eq(EmbDeviceCode::COMPUTE_30);
  check_eq(EmbDeviceCode::COMPUTE_120A);
}

TEST(EmbDeviceCodeTest, getFormat) {
  auto make = [](EmbDeviceCode::Id id) -> EmbDeviceCode::BinaryFormat {
    return EmbDeviceCode(id, elf, "").getBinaryFormat();
  };

  // This checks the first and last device id for each "class" of device. It
  // would be good to keep this up to date with the list of known device id's
  // in EmbDeviceCode::Id, but it is not strictly necessary.
  EXPECT_EQ(make(EmbDeviceCode::GFX600), EmbDeviceCode::AMDGPU);
  EXPECT_EQ(make(EmbDeviceCode::GFX1201), EmbDeviceCode::AMDGPU);
  EXPECT_EQ(make(EmbDeviceCode::SM_30), EmbDeviceCode::NVSASS);
  EXPECT_EQ(make(EmbDeviceCode::SM_120A), EmbDeviceCode::NVSASS);
  EXPECT_EQ(make(EmbDeviceCode::COMPUTE_30), EmbDeviceCode::NVPTX);
  EXPECT_EQ(make(EmbDeviceCode::COMPUTE_120A), EmbDeviceCode::NVPTX);
}

TEST(EmbDeviceCodeTest, getIdForString) {
  auto check_eq = [](StringRef s, EmbDeviceCode::Id expected) -> void {
    Expected<EmbDeviceCode::Id> idOrErr = EmbDeviceCode::getIdFor(s);

    EXPECT_TRUE(bool(idOrErr));
    EXPECT_EQ(*idOrErr, expected);
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

TEST(EmbDeviceCodeTest, getIdForInt) {
  auto check_eq = [](uint64_t n, EmbDeviceCode::Id expected) -> void {
    Expected<EmbDeviceCode::Id> idOrErr = EmbDeviceCode::getIdFor(n);

    EXPECT_TRUE(bool(idOrErr));
    EXPECT_EQ(*idOrErr, expected);
  };

  // These checks will fail when new devices are added. This is intentional.
  check_eq(EmbDeviceCode::AMDGPU_lo + 1, EmbDeviceCode::GFX600);
  check_eq(EmbDeviceCode::AMDGPU_hi - 1, EmbDeviceCode::GFX1201);
  check_eq(EmbDeviceCode::NVSASS_lo + 1, EmbDeviceCode::SM_30);
  check_eq(EmbDeviceCode::NVSASS_hi - 1, EmbDeviceCode::SM_120A);
  check_eq(EmbDeviceCode::NVPTX_lo + 1, EmbDeviceCode::COMPUTE_30);
  check_eq(EmbDeviceCode::NVPTX_hi - 1, EmbDeviceCode::COMPUTE_120A);

  EXPECT_FALSE(EmbDeviceCode::getIdFor(0));
  EXPECT_FALSE(EmbDeviceCode::getIdFor(EmbDeviceCode::AMDGPU_lo));
  EXPECT_FALSE(EmbDeviceCode::getIdFor(EmbDeviceCode::AMDGPU_hi));
  EXPECT_FALSE(EmbDeviceCode::getIdFor(EmbDeviceCode::NVSASS_lo));
  EXPECT_FALSE(EmbDeviceCode::getIdFor(EmbDeviceCode::NVSASS_hi));
  EXPECT_FALSE(EmbDeviceCode::getIdFor(EmbDeviceCode::NVPTX_lo));
  EXPECT_FALSE(EmbDeviceCode::getIdFor(EmbDeviceCode::NVPTX_hi));
  EXPECT_FALSE(EmbDeviceCode::getIdFor(0x8000000000000000));
  EXPECT_FALSE(EmbDeviceCode::getIdFor(0x100010001));
}
