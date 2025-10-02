//===- EmbDeviceCode.cpp - Parse embedded device code in object files -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Parse embedded device code in object files.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/EmbDeviceCode.h"
#include "kitsune/Config/config.h"
#include "kitsune/Core/EmbDeviceCodeUtils.h"
#include "kitsune/Object/BinaryUtils.h"
#include "kitsune/Object/ObjectUtils.h"
#include "kitsune/Support/Error.h"
#include "kitsune/Support/ToString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::object;

static std::string getName(StringRef baseName, EmbDeviceCode::BinaryFormat fmt,
                           StringRef arch, file_magic magic) {
  auto getExt = [](EmbDeviceCode::BinaryFormat fmt,
                   file_magic magic) -> StringRef {
    switch (magic) {
    case file_magic::archive:
      return ".a";
    case file_magic::elf_relocatable:
    case file_magic::macho_object:
      switch (fmt) {
      case EmbDeviceCode::AMDGPU:
        return ".o";
      case EmbDeviceCode::NVSASS:
        return ".cubin";
      case EmbDeviceCode::NVPTX:
        return ".ptx";
      }
      llvm_unreachable("EmbDeviceCode::getName(object): Unknown binary format");
    case file_magic::elf_shared_object:
      return ".so";
    case file_magic::macho_dynamically_linked_shared_lib:
      return ".dylib";
    default:
      llvm_unreachable("EmbDeviceCode::getName: Magic number not handled");
    }
  };

  const auto &[stem, e] = baseName.rsplit('.');
  return sjoin(stem, "-", arch, getExt(fmt, magic));
}

EmbDeviceCode::EmbDeviceCode(StringRef arch, BinaryFormat fmt,
                             MemoryBufferRef inBuf)
    : arch(arch), fmt(fmt) {
  StringRef code = inBuf.getBuffer();
  StringRef baseName = inBuf.getBufferIdentifier();
  file_magic magic = identify_magic(code);

  name = ::getName(baseName, fmt, arch, magic);
  memBuf = MemoryBufferRef(code, name);
}

TTID EmbDeviceCode::getTTID() const {
  switch (getBinaryFormat()) {
  case BinaryFormat::AMDGPU:
    return TTID::Hip;
  case BinaryFormat::NVSASS:
  case BinaryFormat::NVPTX:
    return TTID::Cuda;
  }
  llvm_unreachable("EmbDeviceCode::getTTID: Unknown format");
}

bool EmbDeviceCode::isArchive() const { return object::isArchive(memBuf); }

bool EmbDeviceCode::isObject() const { return object::isObject(memBuf); }

bool EmbDeviceCode::isShared() const { return object::isShared(memBuf); }

Expected<EmbDeviceCode> EmbDeviceCode::create(StringRef arch,
                                              MemoryBufferRef mbuf) {
  if (isAMDGPUArch(arch))
    return EmbDeviceCode(arch, BinaryFormat::AMDGPU, mbuf);
  else if (isNVArch(arch))
    return EmbDeviceCode(arch, BinaryFormat::NVSASS, mbuf);
  else if (isNVVirtArch(arch))
    return EmbDeviceCode(arch, BinaryFormat::NVPTX, mubf);
  else
    createStringError(sjoin(
        "Could not determine binary format from architecture: '", arch, "'"));
}
