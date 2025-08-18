//===- DeviceCodeLinker.cpp - Simple linker for embedded device code ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Linker for embedded device code.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Linker/DeviceCodeLinker.h"
#include "kitsune/Core/EmbDeviceCodeUtils.h"
#include "kitsune/Object/ArchiveUtils.h"
#include "kitsune/Object/ObjectUtils.h"
#include "kitsune/Support/Error.h"
#include "llvm/Object/ObjectFile.h"

using namespace llvm;
using namespace llvm::object;

void LinkedDeviceCode::add(OwningBinary<ObjectFile> objFile) {
  linked.emplace_back(std::move(objFile));
}

Error DeviceCodeLinker::addArchiveTo(const EmbDeviceCode &devCode,
                                     NewArchiveMembers &members) {
  MemoryBufferRef memBuf = devCode.getMemoryBufferRef();
  Expected<std::unique_ptr<Archive>> archive = Archive::create(memBuf);
  if (not archive)
    return archive.takeError();

  Expected<ArchiveMemberObjects> objs = getMemberObjects(**archive);
  if (not objs)
    return objs.takeError();

  for (const std::unique_ptr<ObjectFile> &obj : *objs)
    members.emplace_back(obj->getMemoryBufferRef());

  return Error::success();
}

Error DeviceCodeLinker::addObjectTo(const EmbDeviceCode &devCode,
                                    NewArchiveMembers &members) {
  members.emplace_back(devCode.getMemoryBufferRef());
  return Error::success();
}

Error DeviceCodeLinker::addTo(const EmbDeviceCode &devCode,
                              NewArchiveMembers &members) {
  if (devCode.isArchive())
    return addArchiveTo(devCode, members);
  else if (devCode.isObject())
    return addObjectTo(devCode, members);
  else
    report_internal_error("Unexpected format of embedded device code");
}

Expected<unsigned> DeviceCodeLinker::add(MemoryBufferRef memBuf) {
  return ctx.add(memBuf);
}

Expected<LinkedDeviceCode> DeviceCodeLinker::linkStatic(StringRef triple) {
  LinkedDeviceCode result;
  for (TTID tt : ctx.getTTIDs()) {
    NewArchiveMembers members;
    for (const EmbDeviceCode &devCode : ctx.get(tt))
      if (Error e = addTo(devCode, members))
        return e;

    Expected<std::unique_ptr<MemoryBuffer>> archive =
        writeArchiveToBuffer(members, SymtabWritingMode::NormalSymtab,
                             Archive::getDefaultKind(), /*Deterministic=*/false,
                             /*Thin=*/false);
    RETURN_IF_ERROR(archive);

    std::optional<StringRef> section = getSectionForTTID(tt);
    if (not section)
      report_internal_error("Could not get section to embed device code");

    Expected<OwningBinary<ObjectFile>> wrapperObj = embedIntoNewObject(
        triple, (*archive)->getMemBufferRef(), *section);
    RETURN_IF_ERROR(wrapperObj);

    result.add(std::move(*wrapperObj));
  }
  return result;
}

Expected<LinkedDeviceCode> DeviceCodeLinker::linkFatBinary(StringRef triple) {
  LinkedDeviceCode result;
  llvm_unreachable("NOT YET IMPLEMENTED: linkFatBinary");
  return result;
}
