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
#include "kitsune/Core/EmbDeviceCodeUtils.h"
#include "kitsune/Core/TargetUtils.h"
#include "kitsune/Object/BinaryUtils.h"
#include "kitsune/Object/EmbDeviceCode.h"
#include "kitsune/Object/EmbDeviceCodeContext.h"
#include "kitsune/Support/Error.h"
#include "kitsune/Support/StringUtils.h"
#include "kitsune/Support/ToString.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/ObjCopy/ConfigManager.h"
#include "llvm/ObjCopy/ObjCopy.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::object;

static std::optional<TTID> getTTIDForSection(SectionRef section) {
  if (Expected<StringRef> sectionName = section.getName())
    return getTTIDForSection(*sectionName);
  return std::nullopt;
}

Expected<std::unique_ptr<ObjectFile>>
llvm::object::createObjectFileFrom(StringRef data) {
  return ObjectFile::createObjectFile(MemoryBufferRef(data, ""));
}

Expected<bool> llvm::object::hasSection(const ObjectFile &objFile,
                                        StringRef name) {
  for (SectionRef section : objFile.sections())
    if (Expected<StringRef> sectionName = section.getName())
      if (*sectionName == name)
        return true;
  return false;
}

Expected<bool> llvm::object::hasSymbol(const ObjectFile &objFile,
                                       StringRef name) {
  for (SymbolRef sym : objFile.symbols())
    if (Expected<StringRef> symName = sym.getName())
      if (*symName == name)
        return true;
  return false;
}

Expected<size_t> llvm::object::getNumSections(const ObjectFile &objFile) {
  return std::distance(objFile.section_begin(), objFile.section_end());
}

Expected<size_t> llvm::object::getNumSymbols(const ObjectFile &objFile) {
  return std::distance(objFile.symbol_begin(), objFile.symbol_end());
}

Expected<bool> llvm::object::hasEmbDeviceCode(const ObjectFile &objFile) {
  for (SectionRef sec : objFile.sections())
    if (::getTTIDForSection(sec))
      return true;
  return false;
}

Expected<SmallVector<TTID, 0>>
llvm::object::getEmbDeviceCodeTTIDs(const ObjectFile &objFile) {
  SmallSetVector<TTID, 2> tts;
  for (SectionRef sec : objFile.sections())
    if (std::optional<TTID> tt = ::getTTIDForSection(sec))
      tts.insert(*tt);
  return tts.takeVector();
}

static SmallString<1024> createEmptyObject(StringRef triple) {
  TargetMachine *tm = createTargetMachine(triple.str());
  assert(tm && "Could not get target machine");

  LLVMContext ctx;
  Module m("", ctx);

  std::error_code ec;
  SmallString<1024> buf;
  raw_svector_ostream os(buf);
  legacy::PassManager passMgr;
  if (tm->addPassesToEmitFile(passMgr, os,
                              /*DwoOut=*/nullptr, CodeGenFileType::ObjectFile,
                              /*DisableVerify=*/false))
    report_internal_error("createEmptyObject: ", ec);
  passMgr.run(m);

  return buf;
}

Expected<OwningBinary<ObjectFile>>
llvm::object::embedIntoNewObject(StringRef triple, MemoryBufferRef payload,
                                 StringRef section,
                                 std::optional<StringRef> startSymbol) {
  SmallString<1024> inBuf = createEmptyObject(triple);
  Expected<std::unique_ptr<ObjectFile>> in = createObjectFileFrom(inBuf.str());
  RETURN_IF_ERROR(in);

  objcopy::ConfigManager confMgr;
  objcopy::CommonConfig &conf = confMgr.Common;

  // Strip all symbols and other unnecessary things.
  conf.StripAll = true;

  // StripAll does not remove the .text section, so add that to the list of
  // things to remove explicitly.
  auto errFn = [](Error e) -> Error { return e; };
  if (Error e = conf.ToRemove.addMatcher(objcopy::NameOrPattern::create(
          "[.]text.*", objcopy::MatchStyle::Regex, errFn)))
    return e;

  // Add the section that was requested.
  conf.AddSection.emplace_back(
      section,
      MemoryBuffer::getMemBuffer(payload, /*RequiresNullTerminator=*/false));

  // Set the flags on the new section to be added.
  objcopy::SectionFlag flags = objcopy::SecAlloc | objcopy::SecLoad |
                               objcopy::SecReadonly | objcopy::SecData |
                               objcopy::SecContents;
  conf.SetSectionFlags.try_emplace(section,
                                   objcopy::SectionFlagsUpdate{section, flags});

  // If we have to add a start symbol, do that now. This will generally be
  // required when creating an object file containing a fat binary.
  if (startSymbol) {
    SmallVector<objcopy::SymbolFlag, 0> symFlags = {
        objcopy::SymbolFlag::Global,
        objcopy::SymbolFlag::Protected,
    };
    conf.SymbolsToAdd.emplace_back(objcopy::NewSymbolInfo{
        *startSymbol, section, /*Value=*/0, symFlags, {}});
  }

  SmallString<1024> buf;
  raw_svector_ostream os(buf);
  if (Error e = objcopy::executeObjcopyOnBinary(confMgr, **in, os))
    return e;

  std::unique_ptr<MemoryBuffer> memBuf =
      MemoryBuffer::getMemBufferCopy(buf.str(), "");
  Expected<std::unique_ptr<ObjectFile>> res =
      ObjectFile::createObjectFile(*memBuf);
  if (not res)
    return res.takeError();
  return OwningBinary<ObjectFile>(std::move(*res), std::move(memBuf));
}

static Expected<EmbDeviceCode> parseSection(SectionRef sec) {
  Expected<StringRef> contents = sec.getContents();
  RETURN_IF_ERROR(contents);

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

Expected<unsigned> EmbDeviceCodeContext::add(const ObjectFile &objFile,
                                             StringRef fileName, bool unique) {
  if (unique and bins.contains(fileName))
    return 0;

  unsigned added = 0;
  for (SectionRef sec : objFile.sections()) {
    if (std::optional<TTID> tt = ::getTTIDForSection(sec)) {
      Expected<EmbDeviceCode> devCode = parseSection(sec);
      RETURN_IF_ERROR(devCode);
      devCodes[*tt].push_back(*devCode);
      ++added;
    }
  }

  if (unique and added)
    bins.insert(fileName);
  return added;
}
