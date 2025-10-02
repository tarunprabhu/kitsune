//===- ArchiveUtils.cpp - Utilities for static archives -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for static archives.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/ArchiveUtils.h"
#include "kitsune/Object/BinaryUtils.h"
#include "kitsune/Object/EmbDeviceCode.h"
#include "kitsune/Object/EmbDeviceCodeContext.h"
#include "kitsune/Object/ObjectUtils.h"
#include "kitsune/Support/Error.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::object;

Expected<bool> llvm::object::hasEmbDeviceCode(const Archive &archive) {
  Expected<ArchiveMemberObjects> objs = getMemberObjects(archive);
  RETURN_IF_ERROR(objs);

  for (const std::unique_ptr<ObjectFile> &obj : *objs) {
    Expected<bool> hasDeviceCode = hasEmbDeviceCode(*obj);
    if (not hasDeviceCode)
      return hasDeviceCode.takeError();
    else if (*hasDeviceCode)
      return true;
  }
  return false;
}

Expected<SmallVector<TTID, 0>>
llvm::object::getEmbDeviceCodeTTIDs(const Archive &archive) {
  Expected<ArchiveMemberObjects> objs = getMemberObjects(archive);
  RETURN_IF_ERROR(objs);

  SmallSetVector<TTID, 2> tts;
  for (const std::unique_ptr<ObjectFile> &obj : *objs) {
    Expected<SmallVector<TTID, 0>> objTTs = getEmbDeviceCodeTTIDs(*obj);
    RETURN_IF_ERROR(objTTs);
    for (TTID tt : *objTTs)
      tts.insert(tt);
  }
  return tts.takeVector();
}

Expected<size_t> llvm::object::getNumMembers(const Archive &archive) {
  size_t count = 0;
  Error err = Error::success();
  for ([[gnu::unused]] const Archive::Child &child : archive.children(err)) {
    if (err)
      return err;
    ++count;
  }
  if (err)
    return err;
  return count;
}

Expected<ArchiveMemberObjects>
llvm::object::getMemberObjects(const Archive &archive) {
  ArchiveMemberObjects objFiles;
  Error err = Error::success();
  for (const Archive::Child &child : archive.children(err)) {
    if (err)
      return err;

    Expected<MemoryBufferRef> memBuf = child.getMemoryBufferRef();
    RETURN_IF_ERROR(memBuf);

    Expected<std::unique_ptr<ObjectFile>> objFile =
        ObjectFile::createObjectFile(*memBuf);
    RETURN_IF_ERROR(objFile);

    if (isObject(**objFile))
      objFiles.emplace_back(std::move(*objFile));
  }
  if (err)
    return err;
  return objFiles;
}

Expected<unsigned> EmbDeviceCodeContext::add(const Archive &archive,
                                             StringRef fileName, bool unique) {
  if (unique and bins.contains(fileName))
    return 0;

  Expected<ArchiveMemberObjects> objFiles = getMemberObjects(archive);
  RETURN_IF_ERROR(objFiles);

  EmbDeviceCodeContext archiveCtx;
  for (const std::unique_ptr<ObjectFile> &objFile : *objFiles) {
    Expected<unsigned> r = archiveCtx.add(*objFile);
    RETURN_IF_ERROR(r);
  }

  unsigned added = 0;
  for (const auto &[arch, devCodes] : archiveCtx) {
    SmallVector<NewArchiveMember, 4> newMembers;
    for (const EmbDeviceCode &devCode : devCodes) {
      StringRef name = devCode.getName();
      StringRef code = devCode.getCode();
      NewArchiveMember &member = newMembers.emplace_back();
      member.Buf = MemoryBuffer::getMemBuffer(code, name,
                                              /*RequiresNullTerminator=*/false);
      member.MemberName = name;
    }

    Expected<std::unique_ptr<MemoryBuffer>> newArchive =
        writeArchiveToBuffer(newMembers, SymtabWritingMode::NormalSymtab,
                             Archive::getDefaultKind(), /*Deterministic=*/false,
                             /*Thin=*/false);
    RETURN_IF_ERROR(newArchive);

    StringRef newArchiveCode = (**newArchive).getBuffer();
    MemoryBufferRef newMemBuf(newArchiveCode, fileName);

    Expected<EmbDeviceCode> devCode = EmbDeviceCode::create(arch, newMemBuf);
    RETURN_IF_ERROR(devCode);

    devCodes[arch].push_back(*devCode);
    archives.emplace_back(std::move(*newArchive));
    ++added;
  }

  if (unique and added)
    bins.insert(fileName);
  return added;
}
