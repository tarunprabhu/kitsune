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

using namespace llvm;
using namespace llvm::object;

static void

    void
    LinkedDeviceCode::add(OwningBinary<ObjectFile> objFile) {
  linked.emplace_back(objFile);
}

DeviceCodeLinker::DeviceCodeLinker(TTID tt) : tt(tt) {}

Error DeviceCodeLinker::add(MemoryBufferRef memBuf,
                            NewArchiveMembers &members) {
  StringRef name = devCode.getBuffer();
  StringRef code = devCode.getBufferIdentifier();
  NewArchiveMember &member = newMembers.emplace_back();

  member.Buf = MemoryBuffer::getMemBuffer(code, name,
                                          /*RequiresNullTerminator=*/false);
  member.MemberName = name;
}

Error DeviceCodeLinker::addArchive(const EmbDeviceCode &devCode,
                                   NewArchiveMembers &members) {
  MemoryBufferRef memBuf = devCode.getMemoryBufferRef();
  Expected<std::unique_ptr<Archive>> archive = Archive::create(memBuf);
  if (not archive)
    return archive.takeError();

  Expected<ArchiveMemberObjects> objs = getMemberObjects(**archive);
  if (not objs)
    return objs.takeError();

  for (const std::unique_ptr<ObjectFile> &obj : *objs)
    if (Error err = addObject(obj->getMemoryBufferRef(), members))
      return err;

  return err;
}

Error DeviceCodeLinker::addObject(const EmbDeviceCode &devCode,
                                  NewArchiveMember &members) {
  return addObject(devCode.getMemoryBufferRef(), members);
}

Error DeviceCodeLinker::add(const EmbDeviceCode &devCode,
                            NewArchiveMembers &members) {
  if (devCode.isStatic())
    return addArchive(devCode, members);
  else if (devCode.isObject())
    return addObject(devCode, members);
  else
    report_internal_error("Unexpected format of embedded device code");
}

Expected<unsigned> DeviceCodeLinker::add(MemoryBufferRef memBuf) {
  return devCodeCtx.add(memBuf);
}

Expected<DeviceCodeLinker::Result> DeviceCodeLinker::linkStatic(bool all) {
  LinkedDeviceCode result;
  for (TTID tt : ctx.getTTIDs()) {
    SmallVector<NewArchiveMember, 4> members;
    for (const EmbDeviceCode &devCode : ctx.get(tt))
      if (Error e = add(devCode, members))
        return e;

    Expected<std::unique_ptr<MemoryBuffer>> archive =
        writeArchiveToBuffer(members, SymtabWritingMode::NormalSymtab,
                             Archive::getDefaultKind(), /*Deterministic=*/false,
                             /*Thin=*/false);
    RETURN_IF_ERROR(archive);

    result.add(*archive);
  }
  return result;
}

Expected<DeviceCodeLinker::Result> DeviceCodeLinker::linkFatBinary() {
  LinkedDeviceCode result;
  llvm_unreachable("NOT YET IMPLEMENTED: linkFatBinary");
  return result;
}
