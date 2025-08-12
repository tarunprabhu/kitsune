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
#include "kitsune/Object/ObjectUtils.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::object;

Expected<bool> llvm::object::hasEmbDeviceCode(const Archive &archive, TTID tt) {
  Expected<SmallVector<std::unique_ptr<ObjectFile>, 4>> objsOrErr =
      getMemberObjects(archive);
  if (not objsOrErr)
    return objsOrErr.takeError();
  const SmallVector<std::unique_ptr<ObjectFile>, 4> &objs = *objsOrErr;

  Expected<size_t> sizeOrErr = getNumMembers(archive);
  if (not sizeOrErr)
    return sizeOrErr.takeError();
  size_t size = *sizeOrErr;

  if (objs.size() != size)
    return createStringError("All members of the archive must be objects");

  for (const std::unique_ptr<ObjectFile> &obj : objs)
    if (hasEmbDeviceCode(*obj, tt))
      return true;
  return false;
}

Expected<size_t> llvm::object::getNumMembers(const Archive &archive) {
  size_t count = 0;
  Error err = Error::success();
  for (const Archive::Child &child : archive.children(err)) {
    if (err)
      return err;
    ++count;
  }
  return count;
}

Expected<SmallVector<std::unique_ptr<ObjectFile>, 4>>
llvm::object::getMemberObjects(const Archive &archive) {
  SmallVector<std::unique_ptr<ObjectFile>, 4> objFiles;
  Error err = Error::success();
  for (const Archive::Child &child : archive.children(err)) {
    if (err)
      return err;

    Expected<MemoryBufferRef> memBufOrErr = child.getMemoryBufferRef();
    if (not memBufOrErr)
      return memBufOrErr.takeError();
    const MemoryBufferRef &memBuf = *memBufOrErr;

    Expected<std::unique_ptr<ObjectFile>> objFileOrErr =
        ObjectFile::createObjectFile(memBuf);
    if (not objFileOrErr)
      return objFileOrErr.takeError();
    std::unique_ptr<ObjectFile> objFile = std::move(*objFileOrErr);

    if (isObject(*objFile))
      objFiles.emplace_back(std::move(objFile));
  }
  return objFiles;
}

Error llvm::object::validateEmbDeviceCode(const Archive &archive) {
  Expected<SmallVector<std::unique_ptr<ObjectFile>, 4>> objFilesOrErr =
      getMemberObjects(archive);
  if (not objFilesOrErr)
    return objFilesOrErr.takeError();
  const SmallVector<std::unique_ptr<ObjectFile>, 4> &objFiles = *objFilesOrErr;

  Expected<size_t> sizeOrErr = getNumMembers(archive);
  if (not sizeOrErr)
    return sizeOrErr.takeError();
  size_t size = *sizeOrErr;

  if (objFiles.size() != size)
    return createStringError("All members of archive must be objects");

  if (objFiles.empty())
    return Error::success();

  SmallVector<EmbDeviceCode, 1> devCodes;
  for (const std::unique_ptr<ObjectFile> &objFilePtr : objFiles) {
    const ObjectFile &objFile = *objFilePtr;
    Expected<std::optional<EmbDeviceCode>> devCodeOrErr = parse(objFile, tt);
    if (not devCodeOrErr)
      return devCodeOrErr.takeError();

    if (std::optional<EmbDeviceCode> devCode = *devCodeOrErr)
      devCodes.push_back(*devCode);
  }

  EmbDeviceCode::Id id = devCodes[0].getId();
  for (const EmbDeviceCode &devCode : devCodes)
    if (devCode.getId() != id)
      return createStringError("Inconsistent embedded device code targets");

  return Error::success();
}
