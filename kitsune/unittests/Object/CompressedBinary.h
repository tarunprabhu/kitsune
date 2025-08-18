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
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace object {
class Archive;
class ObjectFile;
} // namespace object
} // namespace llvm

/// The object files here are compressed with zlib and encoded to standard
/// base 64. This wrapper class simply decompresses them and provides a memory
/// buffer reference that is needed by the tests.
class CompressedBinary {
public:
  llvm::StringRef inBuf;        /// The compressed buffer
  uint64_t outSize;             /// The decompressed buffer size
  llvm::SmallString<8> outBuf;  /// The decompressed buffer
  llvm::MemoryBufferRef memBuf; /// The decompressed memory buffer

public:
  CompressedBinary(llvm::StringRef name, llvm::StringRef inBuf,
                   uint64_t outSize);
};

/// An empty static archive i.e. it contains no members.
extern const std::unique_ptr<llvm::object::Archive> arEmpty;

/// A static archive that contains a single object file named empty.o that does
/// not contain any embedded device code.
extern const std::unique_ptr<llvm::object::Archive> arNoDeviceCode;

/// A static archive that contains a single object file named sm_72.o that
/// contains embedded NVIDIA GPU code for the sm_72 architecture.
extern const std::unique_ptr<llvm::object::Archive> arCuda1;

/// A static archive contains a single object file named gfx906.o that contains
/// embedded AMDGPU code for the gfx906.o
extern const std::unique_ptr<llvm::object::Archive> arHip1;

/// A static archive contains two object files named sm_72.o and gfx906.o that
/// contain NVIDIA and AMD GPU code for the sm_72 and gfx906 architectures
/// respectively.
extern const std::unique_ptr<llvm::object::Archive> arMulti;

/// This archive contains an an object file named empty.o and a text file named
/// empty.txt. We require static archives that are operated on by Kitsune to
/// consist exclusively of object files. Most operations on archives that
/// contain at least one non-object file will fail.
extern const std::unique_ptr<llvm::object::Archive> arHetero;

/// A simple dynamically linked ELF executable. This just has an empty main
/// function and nothing else.
extern const std::unique_ptr<llvm::object::ObjectFile> elfExecDynamic;

/// A simple statically linked ELF executable. This was originally written in
/// straight assembly, and the _start function simply exits immediately.
extern const std::unique_ptr<llvm::object::ObjectFile> elfExecStatic;

/// An empty ELF shared object. This does not have any exported symbols.
extern const std::unique_ptr<llvm::object::ObjectFile> elfShared;

/// An empty ELF object.
extern const std::unique_ptr<llvm::object::ObjectFile> elfEmpty;

/// A relocatable ELF object containing embedded NVIDIA GPU code for the sm_72
/// architecture.
extern const std::unique_ptr<llvm::object::ObjectFile> elfCuda;

/// A relocatable ELF object containing embedded AMD GPU code for the gfx906
/// architecture.
extern const std::unique_ptr<llvm::object::ObjectFile> elfHip;

/// A relocatable ELF object containing embedded NVIDIA and AMD GPU code for the
/// sm_72 and gfx906 architectures respectively.
extern const std::unique_ptr<llvm::object::ObjectFile> elfMulti;

/// A relocatable ELF object with visible symbols.
extern const std::unique_ptr<llvm::object::ObjectFile> elfSectsSyms;

/// A simple MachO executable.
extern const std::unique_ptr<llvm::object::ObjectFile> machOExec;

/// An empty MachO object.
extern const std::unique_ptr<llvm::object::ObjectFile> machOObject;

/// An empty MachO dynamic shared object.
extern const std::unique_ptr<llvm::object::ObjectFile> machOShared;

#endif // KITSUNE_UNITTESTS_OBJECT_COMPRESSED_BINARY_H
