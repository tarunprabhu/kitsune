//===- EmbDeviceCodeContext.cpp - Embedded device code context ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Context object into which all the embedded device code can be collected.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/EmbDeviceCodeContext.h"
#include "kitsune/Object/ArchiveUtils.h"
#include "kitsune/Object/BinaryUtils.h"
#include "kitsune/Object/ObjectUtils.h"
#include "kitsune/Support/Error.h"
#include "kitsune/Support/ToString.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"

using namespace llvm;
using namespace llvm::object;

/// Empty vector of device code so we can always return a const reference in
/// EmbDeviceCodeContext::get().
static const SmallVector<EmbDeviceCode, 8> noDeviceCode;

const SmallVectorImpl<EmbDeviceCode> &EmbDeviceCodeContext::get(TTID tt) const {
  decltype(devCodes)::const_iterator it = devCodes.find(tt);
  if (it == devCodes.end())
    return noDeviceCode;
  return it->second;
}

SmallVector<TTID, 0> EmbDeviceCodeContext::getTTIDs() const {
  SmallVector<TTID, 0> tts;
  for (const auto &[tt, embCodes] : devCodes)
    tts.push_back(tt);
  return tts;
}

bool EmbDeviceCodeContext::contains(const Binary &bin) const {
  return bins.contains(bin.getFileName());
}

Expected<EmbDeviceCode::Id>
EmbDeviceCodeContext::getEmbDeviceCodeId(TTID tt) const {
  if (devCodes.empty())
    return createStringError(
        sjoin("no embedded device code for '", tt, "' tapir target"));

  const SmallVectorImpl<EmbDeviceCode> &devCodesForTT = get(tt);
  assert(devCodesForTT.size() && "Array for device codes cannot be empty");

  EmbDeviceCode::Id id = devCodesForTT[0].getId();
  for (const EmbDeviceCode &devCode : devCodesForTT)
    if (devCode.getId() != id)
      return createStringError("inconsistent formats and targets for "
                               "embedded device code in archive");
  return id;
}

Expected<unsigned> EmbDeviceCodeContext::add(const Binary &bin) {
  if (const auto *objFile = dyn_cast<ObjectFile>(&bin))
    return add(*objFile, bin.getFileName(), /*unique=*/true);
  else if (const auto *archive = dyn_cast<Archive>(&bin))
    return add(*archive, bin.getFileName(), /*unique=*/true);
  llvm_unreachable("EmbDeviceCodeContext::add: Unsupported binary file format");
}

Expected<unsigned> EmbDeviceCodeContext::add(MemoryBufferRef memBuf) {
  if (isArchive(memBuf)) {
    Expected<std::unique_ptr<Archive>> archive = Archive::create(memBuf);
    RETURN_IF_ERROR(archive);
    return add(**archive, memBuf.getBufferIdentifier(), /*unique=*/false);
  } else if (isObject(memBuf)) {
    Expected<std::unique_ptr<ObjectFile>> obj =
        ObjectFile::createObjectFile(memBuf);
    RETURN_IF_ERROR(obj);
    return add(**obj, memBuf.getBufferIdentifier(), /*unique=*/false);
  }
  llvm_unreachable("EmbDeviceCodeContext::add: Unsupported binary file format");
}
