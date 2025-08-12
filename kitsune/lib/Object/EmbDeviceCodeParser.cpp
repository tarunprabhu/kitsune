//===- EmbDeviceCodeParser.cpp - Parse device code in object files --------===//
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

#include "kitsune/Object/EmbDeviceCodeParser.h"
#include "kitsune/Core/SingletonUtils.h"
#include "kitsune/Linker/DeviceCodeLinker.h"
#include "kitsune/Support/StringUtils.h"
#include "kitsune/Support/ToString.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/ObjectFile.h"

#include <map>

using namespace llvm;
using namespace llvm::object;

static bool isDeviceCodeSection(SectionRef sec, TTID tt) {
  if (Expected<StringRef> name = sec.getName())
    return StringSwitch<bool>(*name)
        .Case(KITSUNE_CUDA_CODE_SECTION, tt == TTID::Cuda)
        .Case(KITSUNE_HIP_CODE_SECTION, tt == TTID::Hip)
        .Default(false);
  return false;
}

static Expected<EmbDeviceCode> parseSection(SectionRef sec) {
  Expected<StringRef> contents = sec.getContents();
  if (not contents)
    return createStringError("Could not parse device code section contents");

  size_t pos = 0;
  const char *buf = contents->data();
  const ObjectFile &objFile = *sec.getObject();

  // The section is guaranteed to contain a single binary blob. If the parent
  // object file is a relocatable object, the blob will also be a relocatable
  // ELF object. If the parent object file is a dynamic shared object (DSO),
  // this will be a static archive. The first 8 bytes of the section is an
  // EmbDeviceCode::Id. The rest of the section is the blob.
  //
  //   struct {
  //     uint64_t id;  // The EmbDeviceCode::Id
  //     byte code[];  // The embedded code.
  //   };
  //
  if (sec.getSize() <= 8)
    return createStringError(sjoin("Expected id at ", pos));
  uint64_t nid = *reinterpret_cast<const uint64_t *>(&buf[pos]);
  Expected<EmbDeviceCode::Id> id = EmbDeviceCode::getIdFor(nid);
  if (not id)
    return createStringError(
        sjoin("Invalid id for embedded device code: ", nid));

  pos += 8;
  StringRef code(&buf[pos], sec.getSize() - pos);

  // The contents of the section must be something that is expected.
  file_magic embMagic = identify_magic(code);
  file_magic srcMagic = identify_magic(objFile.getData());
  switch (file_magic::Impl(embMagic)) {
  case file_magic::elf_relocatable:
    if (srcMagic != file_magic::elf_relocatable)
      return createStringError("Embedded device code in relocatable ELF object "
                               "must also be relocatable ELF object");
    break;
  case file_magic::archive:
    if (srcMagic != file_magic::elf_shared_object)
      return createStringError("Embedded device code in dynamic shared object "
                               "must be a static archive");
    break;
  default:
    return createStringError(
        sjoin("Unexpected magic in embedded device code: ", embMagic));
  }

  return EmbDeviceCode(*id, code, sec.getObject()->getFileName());
}

std::optional<EmbDeviceCode> EmbDeviceCodes::get(TTID tt) const {
  for (const EmbDeviceCode &devCode : codes)
    if (devCode.getTTID() == tt)
      return devCode;
  return std::nullopt;
}

