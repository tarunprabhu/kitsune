//===- ObjectUtils.cpp - Utilities for LLVM's object files ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's object files.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/ObjectUtils.h"
#include "kitsune/Config/config.h"
#include "kitsune/Object/BinaryUtils.h"
#include "kitsune/Object/EmbDeviceCode.h"
#include "kitsune/Object/EmbDeviceCodeContext.h"
#include "kitsune/Support/Error.h"
#include "kitsune/Support/StringUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::object;

/// Get the TTID that generates the section.
static std::optional<TTID> getSectionTTID(const SectionRef &sec) {
  if (Expected<StringRef> name = sec.getName()) {
    return StringSwitch<std::optional<TTID>>(*name)
        .Case(KITSUNE_CUDA_CODE_SECTION, TTID::Cuda)
        .Case(KITSUNE_HIP_CODE_SECTION, TTID::Hip)
        .Default(std::nullopt);
  }
  return std::nullopt;
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
    return createStringError(
        sjoin("corrupt embedded device code section (", sec.getSize(), ")"));
  uint64_t nid = *reinterpret_cast<const uint64_t *>(&buf[pos]);

  pos += 8;
  StringRef code(&buf[pos], sec.getSize() - pos);

  // The contents of the section must be something that is expected.
  if (isObject(objFile)) {
    if (not isObject(code))
      return createStringError("embedded device code in relocatable object "
                               "must be a relocatable object");
  } else if (isShared(objFile)) {
    if (not isArchive(code))
      return createStringError("embedded device code in dynamic shared object "
                               "must be a static archive");
  } else {
    report_internal_error("Section in unexpected kind of binary object");
  }

  MemoryBufferRef memBuf(code, sys::path::filename(objFile.getFileName()));
  return EmbDeviceCode::create(nid, memBuf);
}

Expected<bool> llvm::object::hasEmbDeviceCode(const ObjectFile &objFile) {
  for (SectionRef sec : objFile.sections())
    if (getSectionTTID(sec))
      return true;
  return false;
}

Expected<SmallVector<TTID, 0>>
llvm::object::getEmbDeviceCodeTTIDs(const ObjectFile &objFile) {
  SmallSetVector<TTID, 2> tts;
  for (SectionRef sec : objFile.sections())
    if (std::optional<TTID> tt = getSectionTTID(sec))
      tts.insert(*tt);
  return tts.takeVector();
}

Expected<unsigned> EmbDeviceCodeContext::add(const ObjectFile &objFile) {
  unsigned added = 0;
  for (SectionRef sec : objFile.sections()) {
    if (std::optional<TTID> tt = getSectionTTID(sec)) {
      Expected<EmbDeviceCode> devCode = parseSection(sec);
      if (not devCode)
        return devCode.takeError();
      devCodes[*tt].push_back(*devCode);
      ++added;
    }
  }
  if (added)
    bins.insert(objFile.getFileName());
  return added;
}
