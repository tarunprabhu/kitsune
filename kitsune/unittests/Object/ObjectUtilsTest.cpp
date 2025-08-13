//===- ObjectUtilsTest.cpp - Tests for Kitsune's object file utilities ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/ObjectUtils.h"
#include "CheckUtils.h"
#include "CompressedBinary.h"
#include "llvm/Object/ObjectFile.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

static const detail::CompressedBinary
    cELFEmpty("eNqrd/VxY2JkZIABRgY7BgQPO3BAYgMAY34B2w==", 64);

static const detail::CompressedBinary cELFCuda(
    "eNqrd/VxY2JkZIABRgY7BgQPATSYEWwHKMkMVMnDwATm1UPMMWaHm7OPoQGLOR5Qwz1YPYAmWI"
    "DNYQPZqFecUVxSVJKYBGTB6MpcJDq+OCMvpYJBL69MLzMvLZ9BLyU1qTQ9Pq0oMTcVLJqcmJOT"
    "XpRYkAHmFRTll+SXVBak0tRonMAeiJkZWLHK/QcCEP0PSv+F0n+gNCWAEWwvJnCA0tFY1CMDbh"
    "z6ZxOpXxiImbDo/wGlDaA0ExRzQPkSUNoIi5ng9MeI27/o4Q4UK8ClXwHKh/mRBUrD3IFIItmZ"
    "JXrJ+SmpesmlKYkMtIoXAWZU/+PyFydUjAlHvF5Ak+BAUwcADjlkjg==",
    1000);

static const detail::CompressedBinary cELFHip(
    "eNqrd/VxY2JkZIABRgY7BgQPATKYEWwHKMkMVCkEVV0PMceBGW7OA6zmcEAF9VkR5nAAVbIDWd"
    "5ArADEjr4u7gGhDAzN6xJzUzKKE/WyU4vyUnOKJ6yF8ksSi9JTS3YCeenJebpAShcioaubnlZh"
    "aWAG01eWWlScmZ83iZGJASfQK0mtKGHQg9ipl15QFJ+bWJGZW5pbzKCXl1+SqufuF6pbXJKYnM"
    "2gl5NTlhufmJICNDUdIsugV1xSVJKYBKQrc0E0pQAUJsxYxHdA6QA0cUYsfBBmw2IuNsCCxmfH"
    "YiY+/ehqZSjUbwV1AxMO/QkE3K8DCj+f//lAqgFb+CH7E5v9HljsBoEFUFoCLZw50MQZ9LIzS/"
    "SS81NS9TIyC+DJg2LAjyNdBEAFxYlMF7jCVQDNcA40dQDReVAZ",
    1064);

static const detail::CompressedBinary cELFMulti(
    "eNq1U71O40AQXju/EhIgKKCgSIUo8IJAQkABRoID6QCloaIIi72xrfhP9iYKFZF4gdO1NPcCdz"
    "U0PMM9Ao9Bwd/aGcfBeBWkiE9af57dnZnd2W+uD45/yJKEEkhoB6VWiqNy+q/Ct8h3TiI5tq77"
    "cdYrgzgPqJcXB4LXS3UeYTOOU44y4tAMWcDIJf9L+MoZ4kZounoXYbeDLbfpIazTy7bRaAbEof"
    "GsRmzbCIhvxpYfeMxjVz791tBC7PJRQKXctTeOiF+BX4CfgceBFOf9DBX4PGf/MCYE/rdf9J/l"
    "Q87xfwJeBZZhVMGeB17LiRlhSRLfN1t3PueL/GtgJ3csAifnmIGIoGe1MMjzmHuuKkyulNK+qP"
    "KdUR/8hHx7J/uH9TOEbv4RRzdDgls0cKkd/voLNiOBQdk9twzNVTgp/QVFMZrdrdWNxK9Dg9Dy"
    "3N+SLBYAZrTLEO7nxIYfNBzStZy2E3Lteoziw9MzJWREayFs2x2nQXSdRzX6q586ZFyoAj3dAd"
    "dHvKcEoyzQcxbFjF0R6ElFX9PTwpj+23AGWeB/MeL8y1H9jt88Tr28+g3fMy//kaAf/2T6Lqlz"
    "th8RblkMa55OsWn5qTwGs1pbJ2h8TAl0UoOHXxxR5zmYE9X5f2ahKtBZ1n+68JFF/u/vHK8o",
    1864);

class ObjectUtilsTest : public testing::Test {
protected:
  std::unique_ptr<ObjectFile> elfEmpty;
  std::unique_ptr<ObjectFile> elfCuda;
  std::unique_ptr<ObjectFile> elfHip;
  std::unique_ptr<ObjectFile> elfMulti;

protected:
  template <typename T>
  std::unique_ptr<T> getIfOk(Expected<std::unique_ptr<T>> optr) {
    if (not optr)
      ADD_FAILURE();
    return std::move(*optr);
  }

  ObjectUtilsTest() {
    elfEmpty = getIfOk(ObjectFile::createObjectFile(cELFEmpty.memBuf));
    elfCuda = getIfOk(ObjectFile::createObjectFile(cELFCuda.memBuf));
    elfHip = getIfOk(ObjectFile::createObjectFile(cELFHip.memBuf));
    elfMulti = getIfOk(ObjectFile::createObjectFile(cELFMulti.memBuf));
  }
};

TEST_F(ObjectUtilsTest, hasEmbDeviceCode) {
  detail::check_false(hasEmbDeviceCode(*elfEmpty));
  detail::check_true(hasEmbDeviceCode(*elfCuda));
  detail::check_true(hasEmbDeviceCode(*elfHip));
  detail::check_true(hasEmbDeviceCode(*elfMulti));
}

TEST_F(ObjectUtilsTest, getEmbDeviceCodeTTIDs) {
  using Vec = SmallVector<TTID, 0>;

  detail::check_eq(getEmbDeviceCodeTTIDs(*elfEmpty), Vec({}));
  detail::check_eq(getEmbDeviceCodeTTIDs(*elfCuda), Vec({TTID::Cuda}));
  detail::check_eq(getEmbDeviceCodeTTIDs(*elfHip), Vec({TTID::Hip}));

  Expected<SmallVector<TTID, 0>> tts = getEmbDeviceCodeTTIDs(*elfMulti);
  EXPECT_TRUE(bool(tts));

  std::sort(tts->begin(), tts->end());
  EXPECT_EQ(*tts, Vec({TTID::Cuda, TTID::Hip}));
}
