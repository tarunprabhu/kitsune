//===- CompressedBinary.h - Compressed binary objects for unit tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For the object-related unit tests, we compress a binary, then encode it in
// base64 so the raw object data can be included in the source files. This
// provides utilities to decompress such objects.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_UNITTESTS_OBJECT_COMPRESSED_BINARY_H
#define KITSUNE_UNITTESTS_OBJECT_COMPRESSED_BINARY_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include "gtest/gtest.h"

namespace llvm {

namespace detail {

/// The object files here are compressed with zlib and encoded to standard
/// base 64. This wrapper class simply decompresses them and provides a memory
/// buffer reference that is needed by the tests.
class CompressedBinary {
public:
  StringRef inBuf;        /// The compressed buffer
  uint64_t outSize;       /// The decompressed buffer size
  SmallString<8> outBuf;  /// The decompressed buffer
  MemoryBufferRef memBuf; /// The decompressed memory buffer

public:
  CompressedBinary(StringRef inBuf, uint64_t outSize)
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

} // namespace detail

} // namespace llvm

#endif // KITSUNE_UNITTESTS_OBJECT_COMPRESSED_BINARY_H
