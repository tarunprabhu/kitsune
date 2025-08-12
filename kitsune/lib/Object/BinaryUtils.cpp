//===- BinaryUtils.cpp - Utilities for LLVM's binary objects --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's binary objects
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/BinaryUtils.h"
#include "kitsune/Object/ArchiveUtils.h"
#include "kitsune/Object/ObjectUtils.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::object;

Expected<bool> llvm::object::hasEmbDeviceCode(const Binary &bin, TTID tt) {
  if (auto *objFile = dyn_cast<ObjectFile>(&bin))
    return hasEmbDeviceCode(*objFile, tt);
  else if (auto *archive = dyn_cast<Archive>(&bin))
    return hasEmbDeviceCode(*archive, tt);
  return false;
}

bool llvm::object::isArchive(StringRef data) {
  switch (identify_magic(data)) {
  case file_magic::archive:
    return true;
  default:
    return false;
  }
}

bool llvm::object::isArchive(const MemoryBufferRef &memRef) {
  return isArchive(memRef.getBuffer());
}

bool llvm::object::isArchive(const Binary &bin) {
  return isArchive(bin.getMemoryBufferRef());
}

bool llvm::object::isExecutable(StringRef data) {
  switch (identify_magic(data)) {
  case file_magic::elf_executable:
  case file_magic::macho_executable:
  case file_magic::pecoff_executable:
    return true;
  default:
    return false;
  }
}

bool llvm::object::isExecutable(const MemoryBufferRef &memRef) {
  return isExecutable(memRef.getBuffer());
}

bool llvm::object::isExecutable(const Binary &bin) {
  return isExecutable(bin.getMemoryBufferRef());
}

bool llvm::object::isObject(StringRef data) {
  switch (identify_magic(data)) {
  case file_magic::elf_relocatable:
  case file_magic::macho_object:
  case file_magic::coff_object:
  case file_magic::goff_object:
  case file_magic::spirv_object:
  case file_magic::xcoff_object_32:
  case file_magic::xcoff_object_64:
  case file_magic::wasm_object:
    return true;
  default:
    return false;
  }
}

bool llvm::object::isObject(const MemoryBufferRef &memRef) {
  return isObject(memRef.getBuffer());
}

bool llvm::object::isObject(const Binary &bin) {
  return isObject(bin.getMemoryBufferRef());
}

bool llvm::object::isShared(StringRef data) {
  switch (identify_magic(data)) {
  case file_magic::elf_shared_object:
  case file_magic::macho_dynamically_linked_shared_lib:
  case file_magic::coff_import_library:
    return true;
  default:
    return false;
  }
}

bool llvm::object::isShared(const MemoryBufferRef &memRef) {
  return isShared(memRef.getBuffer());
}

bool llvm::object::isShared(const Binary &bin) {
  return isShared(bin.getMemoryBufferRef());
}
