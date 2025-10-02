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

Expected<unsigned> DeviceCodeLinker::add(MemoryBufferRef memBuf) {
  return ctx.add(memBuf);
}

using NewArchiveMembers = SmallVector<NewArchiveMember, 8>;

static Error addArchiveTo(const EmbDeviceCode &devCode,
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

static Error addObjectTo(const EmbDeviceCode &devCode,
                         NewArchiveMembers &members) {
  members.emplace_back(devCode.getMemoryBufferRef());
  return Error::success();
}

static Error addTo(const EmbDeviceCode &devCode, NewArchiveMembers &members) {
  if (devCode.isArchive())
    return addArchiveTo(devCode, members);
  else if (devCode.isObject())
    return addObjectTo(devCode, members);
  else
    report_internal_error("Unexpected format of embedded device code");
}

Expected<OwningBinary<ObjectFile>>
DeviceCodeLinker::linkStatic(StringRef triple) {
  Expected<OwningBinary<ObjectFile>> result = createEmptyObject(triple);
  if (not result)
    return result.takeError();

  for (const auto &[arch, devCodes] : ctx) {
    NewArchiveMembers members;
    for (const EmbDeviceCode &devCode : ctx)
      if (Error e = addTo(devCode, members))
        return e;

    Expected<std::unique_ptr<MemoryBuffer>> archive =
        writeArchiveToBuffer(members, SymtabWritingMode::NormalSymtab,
                             Archive::getDefaultKind(), /*Deterministic=*/false,
                             /*Thin=*/false);
    RETURN_IF_ERROR(archive);

    std::optional<StringRef> section = getSectionForArch(arch);
    if (not section)
      report_internal_error("Could not get section to embed device code");

    result = embedIntoObject(std::move(*result), std::move(*archive), *section);
    if (not result)
      return result.takeError();
  }
  return result;
}

Expected<OwningBinary<ObjectFile>>
DeviceCodeLinker::linkFatBinary(StringRef triple) {
  OwningBinary<ObjectFile> result;
  llvm_unreachable("NOT YET IMPLEMENTED: linkFatBinary");
  return result;
}

Expected<OwningBinary<ObjectFile>>
DeviceCodeLinker::link(DeviceCodeLinker::Mode mode, StringRef triple) {
  switch (mode) {
  case Mode::Static:
    return linkStatic(triple);
  case Mode::FatBin:
    return linkFatBinary(triple);
  }
  llvm_unreachable("DeviceCodeLinker::link: Mode not handled");
}
