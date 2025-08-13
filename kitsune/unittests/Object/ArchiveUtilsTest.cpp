//===- ArchiveUtilsTest.cpp - Tests for Kitsune's archive utilities -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/ArchiveUtils.h"
#include "CheckUtils.h"
#include "CompressedBinary.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

// This is an empty archive i.e. it contains no members.
static const detail::CompressedBinary cNoMembers("eNpTtEksSs6w4wIACn4CRA==", 8);

// This archive contains a single object file named empty.o that does not
// contain any embedded device code.
static const detail::CompressedBinary cNoDeviceCode(
    "eNpTtEksSs6w49JXQAUGGGwDFHELuGwCFwMUpOYWlFTq5esTMsPMxARMGxlYwM2od/VxY2JkhB"
    "nFwMhgx4DgIUAAEtsBSjKBVOoVlxSVJCYxUAmA7GbGIu4ApTmxqEcGAMBPGhs=",
    344);

// This archive contains a single object file named sm_72.o that contains
// embedded NVIDIA GPU code for the sm_72 architecture.
static const detail::CompressedBinary cCuda(
    "eNpTtEksSs6w49JXQAUGGGwDFHELuGwCFwMUFOfGmxvp5esTMsPMxARMGxoYGMDMqHf1cWNiZI"
    "QZxcDIYMeA4CGABjOC7QAlmYEqeRiYwDyoOcbscHP2MTRgMccDargHqwfQBAuwOWwgG/WKM4pL"
    "ikoSk4AsGF2Zi0THF2fkpVQw6OWV6WXmpeUz6KWkJpWmx6cVJeamgkWTE3Ny0osSCzLAvIKi/J"
    "L8ksqCVJoajRPYAzEzAytWuf9AAKL/Qem/UPoPlKYEMILtxQQOUDoai3pkwI1D/2wi9QsDMRMW"
    "/T+gtAGUZoJiDihfAkobYTETnP4YcfsXPdyBYgW49CtA+TA/skBpmDsQSSQ7s0QvOT8lVS+5NC"
    "WRgVbxIsCM6n9c/uKEijHhiNcLaBIcaOoAPJ54rw==",
    1136);

// This archive contains a single object file named gfx906.o that contains
// embedded AMDGPU code for the gfx906.o
static const detail::CompressedBinary cHip(
    "eNpTtEksSs6w49JXQAUGGGwDFHELuGwCFwMUpKdVWBqY6eXrEzDDzMQETBsamJnAzKh39XFjYm"
    "SEGcXAyGDHgOAhQAYzgu0AJZmBKoWgqqHmODDDzXmA1RwOqKA+K8IcDqBKdiDLG4gVgNjR18U9"
    "IJSBoXldYm5KRnGiXnZqUV5qTvGEtVB+SWJRemrJTiAvPTlPF0jpQiR0dSFBAdNXllpUnJmfN4"
    "mRiQEn0CtJrShh0IPYqZdeUBSfm1iRmVuaW8ygl5dfkqrn7heqW1ySmJzNoJeTU5Ybn5iSAjQ1"
    "HSLLoFdcUlSSmASkK3NBNKUAFCbMWMR3QOkANHFGLHwQZsNiLjbAgsZnx2ImPv3oamUo1G8FdQ"
    "MTDv0JBNyvAwo/n//5QKoBW/gh+xOb/R5Y7AaBBVBaAi2cOdDEGfSyM0v0kvNTUvUyMgvgyYNi"
    "wI8jXQRABcWJTBe4wlUAzXAONHUArFVkYA==",
    1200);

// This archive contains two object files named sm_72.o and gfx906.o that
// contain NVIDIA and AMD GPU code for the sm_72 and gfx906 architectures
// respectively.
static const detail::CompressedBinary cMulti(
    "eNq1VE9v0zAUd5J1KwOJaRzgwGHcOBA3jKoMhEaQgFVioF524pB5iZtEzT/FbtWdmMQXQFy58A"
    "XgDBc+Ax+Bj8EBGG760rRpTIuqPcn5+T37/Rzbz79bj0hqe/ubjZ1ZM+b6xkx8bzJ6vInAWGjd"
    "38VxYxFHq9nM8K5hGDnHm2eHz1VFyamQgvZR4RV2Wyv6Jnw1MfMKUjMPeO5tTHi+obMKnjaQt2"
    "ttwbCX8ayPVsTMYzzl5ET0cjwNp9BiXuQMEY4G2I+6McIOPem7VjclIc2iNgkCNyWJl3lJGvOY"
    "nyb0Qqml9lg0DdUqx86FjfAP4G/AX4CrmJKtO28m4OuK+dN2WZL/Ycn8a6KpFfk/AQ1AFVod/B"
    "uAuxWcWf0p8v2Wz13EEln+Dvj5HtcA8/8oSqTnc2zHDsV23yHoou5lS5vdv2xflyCmSu71e2mg"
    "XprndocPjFYhEos1otX8X43wJBqxDbOBx9QmPD8qeeoQbNQKnrqYOdKWF3CHT14+PegcIfT2Mw"
    "kdjxHco2lEA/buE/icpC7lX4Xn2pEuQB8P6Pr4KPK8AU2ZH0fvFVV+eZjTIUd4vCZ2k9QKydAP"
    "+yETehBzig9eHemME7uHcBAMQos4jmB1x6NzqrOqmZJa+gLYWVBLCrR1SS2Vba3kb0jeqImWe6"
    "M3V8x/CP8gewvHC/7/zuj8Ds9jAWdV5ze9z6r12xKN+1jSsvycyxo3pS2en0zKY2W7KqmLDgSv"
    "L1kXsnPd0v6tMX8BseTSug==",
    2260);

// This archive contains an an object file named empty.o and a text file named
// empty.txt. This is used to check that we don't require an archive to
// contain object files only.
static const detail::CompressedBinary cHetero(
    "eNpTtEksSs6w49JXQAUGGGwDFHELuGwCFwMUpOYWlFTq5esTMsPMxARMGxlYwM2od/VxY2JkhB"
    "nFwMhgx4DgIUAAEtsBSjKBVOoVlxSVJCYxUAmA7GbGIu4ApTmxqEcGkLAoqSjRJy4sDJHCk4sL"
    "AK9BJQQ=",
    406);

class ArchiveUtilsTest : public testing::Test {
protected:
  std::unique_ptr<Archive> noMembers;
  std::unique_ptr<Archive> noDeviceCode;
  std::unique_ptr<Archive> cuda;
  std::unique_ptr<Archive> hip;
  std::unique_ptr<Archive> multi;
  std::unique_ptr<Archive> hetero;

protected:
  template <typename T>
  std::unique_ptr<T> getIfOk(Expected<std::unique_ptr<T>> optr) {
    if (not optr)
      ADD_FAILURE();
    return std::move(*optr);
  }

  ArchiveUtilsTest() {
    noMembers = getIfOk(Archive::create(cNoMembers.memBuf));
    noDeviceCode = getIfOk(Archive::create(cNoDeviceCode.memBuf));
    cuda = getIfOk(Archive::create(cCuda.memBuf));
    hip = getIfOk(Archive::create(cHip.memBuf));
    multi = getIfOk(Archive::create(cMulti.memBuf));
    hetero = getIfOk(Archive::create(cHetero.memBuf));
  }
};

TEST_F(ArchiveUtilsTest, hasEmbDeviceCode) {
  detail::check_false(hasEmbDeviceCode(*noMembers));
  detail::check_false(hasEmbDeviceCode(*noDeviceCode));
  detail::check_true(hasEmbDeviceCode(*cuda));
}

TEST_F(ArchiveUtilsTest, getEmbDeviceCodeTTIDs) {
  using Vec = SmallVector<TTID, 0>;

  detail::check_eq(getEmbDeviceCodeTTIDs(*noMembers), Vec({}));
  detail::check_eq(getEmbDeviceCodeTTIDs(*noDeviceCode), Vec({}));
  detail::check_eq(getEmbDeviceCodeTTIDs(*cuda), Vec({TTID::Cuda}));
  detail::check_eq(getEmbDeviceCodeTTIDs(*hip), Vec({TTID::Hip}));

  Expected<SmallVector<TTID, 0>> tts = getEmbDeviceCodeTTIDs(*multi);
  EXPECT_TRUE(bool(tts));

  std::sort(tts->begin(), tts->end());
  EXPECT_EQ(*tts, Vec({TTID::Cuda, TTID::Hip}));
}

TEST_F(ArchiveUtilsTest, getNumMembers) {
  detail::check_eq(getNumMembers(*noMembers), 0UL);
  detail::check_eq(getNumMembers(*noDeviceCode), 1UL);
  detail::check_eq(getNumMembers(*cuda), 1UL);
  detail::check_eq(getNumMembers(*hip), 1UL);
  detail::check_eq(getNumMembers(*multi), 2UL);
}

TEST_F(ArchiveUtilsTest, getMemberObjects) {
  auto check = [](const Archive &archive,
                  const SmallSet<StringRef, 2> &fileNames) {
    Expected<ArchiveMemberObjects> objsOrErr = getMemberObjects(archive);
    EXPECT_TRUE(bool(objsOrErr));
    const ArchiveMemberObjects &objs = *objsOrErr;

    EXPECT_EQ(objs.size(), fileNames.size());
    for (const std::unique_ptr<ObjectFile> &obj : objs)
      EXPECT_TRUE(fileNames.contains(obj->getFileName()));
  };

  check(*noMembers, {});
  check(*noDeviceCode, {"empty.o"});
  check(*cuda, {"sm_72.o"});
  check(*hip, {"gfx906.o"});
  check(*multi, {"sm_72.o", "gfx906.o"});
  check(*hetero, {"empty.o", "empty.txt"});
}
