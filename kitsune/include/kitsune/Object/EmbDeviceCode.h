//==- EmbDeviceCode.h - Wrapper object for embedded device code -*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wrapper object for embedded device code. These are generally constructed
// by the embedded device code context.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_OBJECT_EMB_DEVICE_CODE_H
#define KITSUNE_OBJECT_EMB_DEVICE_CODE_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {

/// An embedded device code object. This consists of a reference to the code
/// buffer with metadata about the code itself.
struct EmbDeviceCode {
public:
  /// The format of the device code buffer i.e. what the binary data in the
  /// buffer represents. None of the elements of this enum must be 0.
  enum BinaryFormat {
    AMDGPU = 1, /// AMD GPU machine code
    NVSASS,     /// NVIDIA machine code
    NVPTX,      /// NVIDIA PTX code
  };

private:
  /// The architecture of the embedded device code.
  StringRef arch;

  /// The identifier describing the format of the device code.
  BinaryFormat fmt;

  /// The code.
  MemoryBufferRef memBuf;

  /// The name of the buffer. This is derived from the name of the parent file
  /// in which this was embedded, the binary format and the target architecture.
  std::string name;

private:
  EmbDeviceCode(StringRef arch, BinaryFormat fmt, MemoryBufferRef mbuf);

public:
  /// The tapir target that generated this embedded device code.
  TTID getTTID() const;

  /// Get the memory buffer containing the code.
  MemoryBufferRef getMemoryBufferRef() const { return memBuf; }

  /// Get the code.
  StringRef getCode() const { return memBuf.getBuffer(); }

  /// Get the name of the code buffer.
  StringRef getName() const { return memBuf.getBufferIdentifier(); }

  /// The architecture as a string.
  StringRef getArch() const { return arch; }

  /// Get the format of this embedded device code.
  BinaryFormat getBinaryFormat() const { return fmt; }

  /// Is this embedded device code a static archive.
  bool isArchive() const;

  /// Is this embedded device code a relocatable object file. Dynamic shared
  /// objects (DSO's) are not considered objects by this method. Use
  /// \ref isShared to check for that.
  bool isObject() const;

  /// Is this embedded device code a shared library.
  bool isShared() const;

public:
  /// Construct an embedded device code object.
  ///
  /// \param arch The architecture name for the binary code
  /// \param mbuf A memory buffer containing the device code. The name of this
  ///             buffer is typically the name of the file in which the device
  ///             code was embedded. It is used to derive the final name of the
  ///             the code buffer maintained in this object.
  ///
  static Expected<EmbDeviceCode> create(StringRef arch, MemoryBufferRef mbuf);
};

} // namespace llvm

#endif // KITSUNE_OBJECT_EMB_DEVICE_CODE_H
