//===- BinaryUtilsTest.cpp - Tests for LLVM's binary object utilities -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/BinaryUtils.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Compression.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

struct Compressed {
  StringRef inBuf;        /// The compressed buffer
  uint64_t outSize;       /// The decompressed buffer size
  SmallString<8> outBuf;  /// The decompressed buffer
  MemoryBufferRef memBuf; /// The decompressed memory buffer

  Compressed(StringRef inBuf, uint64_t outSize)
      : inBuf(inBuf), outSize(outSize) {
    outBuf.resize_for_overwrite(outSize);

    DebugCompressionType zlib = DebugCompressionType::Zlib;
    std::vector<char> decoded;
    if (Error err = decodeBase64(inBuf, decoded))
      ADD_FAILURE();
    ArrayRef<uint8_t> in(reinterpret_cast<const uint8_t *>(decoded.data()),
                         decoded.size());
    uint8_t *out = reinterpret_cast<uint8_t *>(outBuf.data());
    if (Error err = compression::decompress(zlib, in, out, outSize))
      ADD_FAILURE();

    memBuf = MemoryBufferRef(StringRef(outBuf.data(), outSize), "");
  }
};

static const StringRef sArchive("!<arch>\n", 8);

// The minimum size of ELF objects at this time is 18.
static const Compressed cELFExec(
    "eNrtmt9LVEEUx+fuD11N260UTCUlRAraXdZfGOEPUvNKGmb64EPeVl1tYV1DVxJ7SLKgRYQC6b"
    "k/IcKXnlpRRN/0rUcNBQOtjQqswNvM3jO7cycXpBcJzkfW7z1nzjkzd+a6eJl53NJ+w6IohGMl"
    "9YRZpS7DbiR/00hqyWm4zqAfmxAnq8tqVkeyHyNvHvyyFhOzKoLaSHp27WYlrlSeXbBl9SlmFf"
    "MS/ZWCX9LnxKxiHpsblwfc9WYlFpgPiznPAnmlkFdab9bXiln5fNrg0wnzJ6s8fDnvAcTJWkbM"
    "apPy7kNBWeX+eN6dncjgv4yzE/Js0CCr/Lzw/m7TvAxyfPhj0QX9pVu/uGJWRejXBc9a660eYp"
    "kmMZvUngfPP2t/P9+7sPdQySy6ub21PLyf0bM20usNBftrqryhQXcoGJ6YdE/W1rhrqjzjo56K"
    "RB1FqHdcWPzZI/xNwlyLaGn84TT+c/Rz8aiONY3ezYA2HvGPRbQRfzBMPQOTfm0oGPaHglMBwp"
    "rZvdWQ1va2601ahafCU528rqwiWlt3hzYYGAsMB8cjgbHujqbQaDjQ7e8PBWit4ZHRMFTXjNAj"
    "A9lflpL4scJ8WBLj5es9URjMYqtyBXwLL19lsJxaYl53/jy6zhhaIPljUNDRaPZzO9aQ+t4U13"
    "BV8FsE/7rgtwr+Dw3m71/OpuC3EwRBEARBEAQx88154Zc6s+dQZ+2LXkLUZ7GIRV9XZ5YdS8l2"
    "335bdKNPjX5UZ7bjnd0tvphvTZ2r26LNev4jmvV9yOMseZqIp/53rM6c/S2TqweRfFq6D0pn6Z"
    "vOkmkWtwRK46OJ+OonTC4fqtG4uvi5QV08sKrKirpxGMmjBdxQwKFvDjlLmlP5bHzTdS1e9i98"
    "ZY86U5fNLtXoTiRXna0r97IhHtB3uN0BXddX7MXUodxdSo2X1mkkQymb1fs0TGOhvi+2lJqj5K"
    "wgCIIgCIIgCIIgyP8F2wdTiqzXCoix71Qc1/Uqpl91vZ0Ye5WJuKkuoky6lKKcTMcLxfCzvdbN"
    "L7pezgKaMxM7X+wXy1+ldcT9Kr4vyva82L4n99lwCRAEQRAEQRAEQU4Efq4xJpxjZvCznTnJQE"
    "NywSyDvPO8Gc5FFoHNz0EWgvJzk8VS+49DfZTpG3h55GcY78GLIj/Luw7t2WDz89eneH3QfOn+"
    "MkEv2c39xkCzpPzfujEeHnoI9gGMRwebz1Mc7F1o/wm29YTXlZ8fTwc/n/oHZVAGWQ==",
    12304);

static const Compressed cELFObject("eNqrd/VxY2JkZIABRgY7BgQPO3BAYgMAY34B2w==",
                                   64);

static const Compressed cELFShared(
    "eNrtms9rE0EUx2fzy41aUqQFrdIU8VChpBCpRCFpsL+m0BYsiXgpS5omYTFNIFlBvBiMCiVUiv"
    "g3iHjz5s2GSOlJol6Kp3goKBQpiND2kHUn+8bMbtOiJw++T9h85715b37t7mHhPZiYmXRIEuE4"
    "SYS0LUKi5DBREiIytFmsixyDy6bd7Ty3YNt1n1hVzGsNNQB+m9qnE/M8xhUKmHYoYtU6bHpLsu"
    "Y5II9CHo1YtSRZVRbmd/EBOuglYlW+3pvb2pKH/Dn82OaNPNcx+9uUrCoJ8/a07jshU3Nx8uRM"
    "Xn3Wf+vTxIvn85WP78mb4MO3PF4S8v5mfQMd/NIR/qEj/JPGdbHTBIqSWc7nlKKWKGiKQpTp2K"
    "yylCqkMmpRSxVis2PZfC4VSyxmU2Zf5x4leS+hpNVcIqveT5GsupgMFPOBq2RqZvrGmBIMBAMj"
    "5qrZzwGtuHD+d/tUL7u5CrGeO38eGhAYtfnXuq0vmmx78TZHTfXYzr4u+B2Cf0vwOwV/Q/C7CI"
    "IgCIIgCIIg/yM/fP0HtLwj04q7Omx81j5e1xx6nZbfyTXzO3c1/IX5V92fmVzb03qN0KcQ6tUb"
    "Pn+JxdVAjfjXrfiRV0wuN+nKLq1+H6XVPSeVNuiHptZjDLAAA8h6I+3zj7fz2XpK4eVh9kl1JU"
    "7L4eusSVe2tS5aCc8Zht47aPx/Teq6vuEeN5rSQi0d8PkfEVhHlKTbNhvvW8aI9fnbe/29OwRB"
    "EARBEARBEARBEARBEARBEARBEITXs64J9dWMOOhpHgj9XWDugH0ObF4nex5sXv/aB8rrZS/Y+n"
    "829TzTQSh+dYN/HWxeQ03BPgn2HdBToGdBT4DedlrneQnqtcU3dXP+EMTrYPNz2AV7CPr3wXb+"
    "o/vF69UPETXlF/CkuGk=",
    12296);

static const Compressed cMachOExec(
    "eNrtmb9LQlEUx8/1qUmZPajBfgyviLBJXeKNRUpKUiIKEcTN8oWCGpiSNbkUNATtLfafBA2N/Q"
    "EtLY3R3lDd6zvizR7l0BTnAwc913e+75xzz+K9D28v70MATAMAlzC/sCfx5QwMkEwKSwjjPL2y"
    "Ft+OZzbhGwx+R+qYTOpk41tZh3ijLwB9D5qGy5zXrWa991i/nqHZej7Fd6u6bmipLueN6nGpWu"
    "Cl6sGhg14I9Uy957t+qJNzq8gPavmK5Zzf7ritt6z4mhI/Ipb6+2b3P5XcWI/Hktlefzp907/2"
    "a1fpqbovcwCtyAD7JHUiOAtBWTe+J2HYvsgP0g77P8gMOBHA1OXchRtHtXC5tBcunJQL+Ps05p"
    "FKxa5zuWjs5l67jDxOLc13S3cPdywk6gvifqss4HMRw56JRfRNY7D8comL5+jtzp2sbwbrNJSW"
    "Y5/c8r0m2BbCHEycW7+I0oEgCIIgCIIgCIIgiP/KBH6y0wywps50/5DvigHI8wt5ntB+/ejQPa"
    "+ClYB+7lod9VLrCIIgCIIgCIL4axgHD7gq+VIVZnmlyK2mtd+oW7xo5QtWDRY0aHuVG+82/jGR"
    "K2PMvtW070S5QzTAJ4BQWDE=",
    8288);

static const Compressed cMachOObject(
    "eNo7/+vtP3YGBkZmBiABxExAvAKIGRQYwEASiGcwEAYn0PjsUMwI5cfHl6RWlCDk4+NDXCNCiD"
    "cPCBqQOSpALABisHCB+QAQpQwj",
    200);

static const Compressed cMachOShared(
    "eNo7/+vtP3YGBkZmBgYGNiDmAuIFjAwMrQwCDCAgCcQeQBwfH+IaEcJANGCFYhhAmOPj6eft6u"
    "IJM0sAlT7AiCrMCMUwoMTA0GBAvDMYmIBYAohPQA0B0RxAmhuIAxioB3iB2ABqFzLQL8kt0E/U"
    "y01MzsjXS6nMyUyCiEtD1fr4uJiFhhq6LIy5Ffj9ezaLCszvLFxgrAblH4C6WxPKP8GIao8ClA"
    "YArX8e8g==",
    488);

class ObjectUtilsTest : public testing::Test {
protected:
  MemoryBufferRef mArchive;

  std::unique_ptr<Archive> archive;
  std::unique_ptr<ObjectFile> elfExec;
  std::unique_ptr<ObjectFile> elfObject;
  std::unique_ptr<ObjectFile> elfShared;
  std::unique_ptr<ObjectFile> machOExec;
  std::unique_ptr<ObjectFile> machOObject;
  std::unique_ptr<ObjectFile> machOShared;

protected:
  template <typename T>
  std::unique_ptr<T> getIfOk(Expected<std::unique_ptr<T>> optr) {
    if (not optr)
      ADD_FAILURE();
    return std::move(*optr);
  }

  ObjectUtilsTest() {
    mArchive = MemoryBufferRef(sArchive, "archive");

    archive = getIfOk(Archive::create(mArchive));
    elfExec = getIfOk(ObjectFile::createObjectFile(cELFExec.memBuf));
    elfObject = getIfOk(ObjectFile::createObjectFile(cELFObject.memBuf));
    elfShared = getIfOk(ObjectFile::createObjectFile(cELFShared.memBuf));
    machOExec = getIfOk(ObjectFile::createObjectFile(cMachOExec.memBuf));
    machOObject = getIfOk(ObjectFile::createObjectFile(cMachOObject.memBuf));
    machOShared = getIfOk(ObjectFile::createObjectFile(cMachOShared.memBuf));
  }
};

TEST_F(ObjectUtilsTest, isArchive) {
  auto check = [](const Compressed &c, bool expected) -> void {
    const MemoryBufferRef &memBuf = c.memBuf;
    StringRef strRef = memBuf.getBuffer();

    EXPECT_EQ(isArchive(memBuf), expected);
    EXPECT_EQ(isArchive(strRef), expected);
  };

  EXPECT_TRUE(isArchive(*archive));
  EXPECT_FALSE(isArchive(*elfExec));
  EXPECT_FALSE(isArchive(*elfObject));
  EXPECT_FALSE(isArchive(*elfShared));
  EXPECT_FALSE(isArchive(*machOExec));
  EXPECT_FALSE(isArchive(*machOObject));
  EXPECT_FALSE(isArchive(*machOShared));

  // FIXME: Also encode archives in base64.
  // check(mArchive, true);
  check(cELFExec, false);
  check(cELFObject, false);
  check(cELFShared, false);
  check(cMachOExec, false);
  check(cMachOObject, false);
  check(cMachOShared, false);
}

TEST_F(ObjectUtilsTest, isExecutable) {
  auto check = [](const Compressed &c, bool expected) -> void {
    const MemoryBufferRef &memBuf = c.memBuf;
    StringRef strRef = memBuf.getBuffer();

    EXPECT_EQ(isExecutable(memBuf), expected);
    EXPECT_EQ(isExecutable(strRef), expected);
  };

  EXPECT_FALSE(isExecutable(*archive));
  EXPECT_TRUE(isExecutable(*elfExec));
  EXPECT_FALSE(isExecutable(*elfObject));
  EXPECT_FALSE(isExecutable(*elfShared));
  EXPECT_TRUE(isExecutable(*machOExec));
  EXPECT_FALSE(isExecutable(*machOObject));
  EXPECT_FALSE(isExecutable(*machOShared));

  // check(mArchive, false);
  check(cELFExec, true);
  check(cELFObject, false);
  check(cELFShared, false);
  check(cMachOExec, true);
  check(cMachOObject, false);
  check(cMachOShared, false);
}

TEST_F(ObjectUtilsTest, isObject) {
  auto check = [](const Compressed &c, bool expected) -> void {
    const MemoryBufferRef &memBuf = c.memBuf;
    StringRef strRef = memBuf.getBuffer();

    EXPECT_EQ(isObject(memBuf), expected);
    EXPECT_EQ(isObject(strRef), expected);
  };

  EXPECT_FALSE(isObject(*archive));
  EXPECT_FALSE(isObject(*elfExec));
  EXPECT_TRUE(isObject(*elfObject));
  EXPECT_FALSE(isObject(*elfShared));
  EXPECT_FALSE(isObject(*machOExec));
  EXPECT_TRUE(isObject(*machOObject));
  EXPECT_FALSE(isObject(*machOShared));

  // check(mArchive, false);
  check(cELFExec, false);
  check(cELFObject, true);
  check(cELFShared, false);
  check(cMachOExec, false);
  check(cMachOObject, true);
  check(cMachOShared, false);
}

TEST_F(ObjectUtilsTest, isShared) {
  auto check = [](const Compressed &c, bool expected) -> void {
    const MemoryBufferRef &memBuf = c.memBuf;
    StringRef strRef = memBuf.getBuffer();

    EXPECT_EQ(isShared(memBuf), expected);
    EXPECT_EQ(isShared(strRef), expected);
  };

  EXPECT_FALSE(isShared(*archive));
  EXPECT_FALSE(isShared(*elfExec));
  EXPECT_FALSE(isShared(*elfObject));
  EXPECT_TRUE(isShared(*elfShared));
  EXPECT_FALSE(isShared(*machOExec));
  EXPECT_FALSE(isShared(*machOObject));
  EXPECT_TRUE(isShared(*machOShared));

  // check(mArchive, false);
  check(cELFExec, false);
  check(cELFObject, false);
  check(cELFShared, true);
  check(cMachOExec, false);
  check(cMachOObject, false);
  check(cMachOShared, true);
}
