//===- EmbDeviceCode.cpp - Parse embedded device code in object files -----===//
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

#include "kitsune/Object/EmbDeviceCode.h"
#include "kitsune/Config/config.h"
#include "kitsune/Object/BinaryUtils.h"
#include "kitsune/Object/ObjectUtils.h"
#include "kitsune/Support/Error.h"
#include "kitsune/Support/StringUtils.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Path.h"

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

static Expected<std::optional<EmbDeviceCode>> parseSection(SectionRef sec) {
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
  if (isObject(objFile)) {
    if (not isObject(code))
      return createStringError("Embedded device code in relocatable object "
                               "must be a relocatable object");
  } else if (isShared(objFile)) {
    if (not isArchive(code))
      return createStringError("Embedded device code in dynamic shared object "
                               "must be a static archive");
  } else {
    report_internal_error("Section in unexpected kind of binary object");
  }

  llvm_unreachable("NOT YET IMPLEMENTED: parseSection");
  // return EmbDeviceCode(objFile, *id, code, objFile.getFileName());
}

static StringRef getExt(EmbDeviceCode::BinaryFormat fmt, file_magic magic) {
  switch (magic) {
  case file_magic::archive:
    return ".a";
  case file_magic::elf_relocatable:
  case file_magic::macho_object:
    switch (fmt) {
    case EmbDeviceCode::AMDGPU:
      return ".o";
    case EmbDeviceCode::NVSASS:
      return ".cubin";
    case EmbDeviceCode::NVPTX:
      return ".ptx";
    }
    llvm_unreachable("getExt(elf_relocatable): Unknown binary format");
  case file_magic::elf_shared_object:
    return ".so";
  case file_magic::macho_dynamically_linked_shared_lib:
    return ".dylib";
  default:
    outs() << "magic: " << magic << "\n";
    llvm_unreachable("getExt: Magic number not handled");
  }
}

EmbDeviceCode::EmbDeviceCode(const Binary &bin, Id id, StringRef code,
                             StringRef hostFileName)
    : bin(bin), id(id), code(code) {
  file_magic magic = identify_magic(code);
  BinaryFormat fmt = getBinaryFormat();
  StringRef ext = getExt(fmt, magic);
  const auto &[base, e] = sys::path::filename(hostFileName).rsplit('.');

  name = sjoin(base, "-", getArch(), ext);
}

TTID EmbDeviceCode::getTTID() const {
  switch (getBinaryFormat()) {
  case BinaryFormat::AMDGPU:
    return TTID::Hip;
  case BinaryFormat::NVSASS:
  case BinaryFormat::NVPTX:
    return TTID::Cuda;
  }
  llvm_unreachable("EmbDeviceCode::getTTID: Unknown format");
}

StringRef EmbDeviceCode::getArch() const {
  // clang-format off
  switch (id) {
    // AMDGPU architecture
  case GFX600: return "gfx600";
  case GFX601: return "gfx601";
  case GFX602: return "gfx602";
  case GFX700: return "gfx700";
  case GFX701: return "gfx701";
  case GFX702: return "gfx702";
  case GFX703: return "gfx703";
  case GFX704: return "gfx704";
  case GFX705: return "gfx705";
  case GFX801: return "gfx801";
  case GFX802: return "gfx802";
  case GFX803: return "gfx803";
  case GFX805: return "gfx805";
  case GFX810: return "gfx810";
  case GFX900: return "gfx900";
  case GFX902: return "gfx902";
  case GFX904: return "gfx904";
  case GFX906: return "gfx906";
  case GFX908: return "gfx908";
  case GFX90A: return "gfx90a";
  case GFX90C: return "gfx90c";
  case GFX940: return "gfx940";
  case GFX941: return "gfx941";
  case GFX942: return "gfx942";
  case GFX950: return "gfx950";
  case GFX1010: return "gfx1010";
  case GFX1011: return "gfx1011";
  case GFX1012: return "gfx1012";
  case GFX1013: return "gfx1013";
  case GFX1030: return "gfx1030";
  case GFX1031: return "gfx1031";
  case GFX1032: return "gfx1032";
  case GFX1033: return "gfx1033";
  case GFX1034: return "gfx1034";
  case GFX1035: return "gfx1035";
  case GFX1036: return "gfx1036";
  case GFX1100: return "gfx1100";
  case GFX1101: return "gfx1101";
  case GFX1102: return "gfx1102";
  case GFX1103: return "gfx1103";
  case GFX1150: return "gfx1150";
  case GFX1151: return "gfx1151";
  case GFX1152: return "gfx1152";
  case GFX1153: return "gfx1153";
  case GFX1200: return "gfx1200";
  case GFX1201: return "gfx1201";

    // NVIDIA cuda architecture
  case SM_30: return "sm_30";
  case SM_32: return "sm_32";
  case SM_35: return "sm_35";
  case SM_37: return "sm_37";
  case SM_50: return "sm_50";
  case SM_52: return "sm_52";
  case SM_53: return "sm_53";
  case SM_60: return "sm_60";
  case SM_61: return "sm_61";
  case SM_62: return "sm_62";
  case SM_70: return "sm_70";
  case SM_72: return "sm_72";
  case SM_75: return "sm_75";
  case SM_80: return "sm_80";
  case SM_86: return "sm_86";
  case SM_87: return "sm_87";
  case SM_89: return "sm_89";
  case SM_90: return "sm_90";
  case SM_90A: return "sm_90a";
  case SM_100: return "sm_100";
  case SM_100A: return "sm_100a";
  case SM_101: return "sm_101";
  case SM_101A: return "sm_101a";
  case SM_120: return "sm_120";
  case SM_120A: return "sm_120a";

    // NVIDIA PTX virtual architecture
  case COMPUTE_30: return "compute_30";
  case COMPUTE_32: return "compute_32";
  case COMPUTE_35: return "compute_35";
  case COMPUTE_37: return "compute_37";
  case COMPUTE_50: return "compute_50";
  case COMPUTE_52: return "compute_52";
  case COMPUTE_53: return "compute_53";
  case COMPUTE_60: return "compute_60";
  case COMPUTE_61: return "compute_61";
  case COMPUTE_62: return "compute_62";
  case COMPUTE_70: return "compute_70";
  case COMPUTE_72: return "compute_72";
  case COMPUTE_75: return "compute_75";
  case COMPUTE_80: return "compute_80";
  case COMPUTE_86: return "compute_86";
  case COMPUTE_87: return "compute_87";
  case COMPUTE_89: return "compute_89";
  case COMPUTE_90: return "compute_90";
  case COMPUTE_90A: return "compute_90a";
  case COMPUTE_100: return "compute_100";
  case COMPUTE_100A: return "compute_100a";
  case COMPUTE_101: return "compute_101";
  case COMPUTE_101A: return "compute_101a";
  case COMPUTE_120: return "compute_120";
  case COMPUTE_120A: return "compute_120a";

    // Sentinels
  case INVALID:
  case AMDGPU_lo:
  case AMDGPU_hi:
  case NVSASS_lo:
  case NVSASS_hi:
  case NVPTX_lo:
  case NVPTX_hi:
    llvm_unreachable("EmbDeviceCode::getArch: Got sentinel");
  }

  llvm_unreachable("EmbDeviceCode::getArch: DeviceID not handled");
  // clang-format on
}

Expected<EmbDeviceCode::Id> EmbDeviceCode::getIdFor(StringRef s) {
  // We could have used a StringSwitch here, but we use a StringMap instead
  // because the StringSwitch<Expected<EmbDeviceCode::Id>> is unreliable.
  // Depending on the compiler, the Expected<.> object may be copied instead of
  // being moved. When this happens, the destructor raises a runtime error
  // because the destructed object has not been checked for an error.
  static const StringMap<EmbDeviceCode::Id> archs = {
      {"gfx600", EmbDeviceCode::GFX600},
      {"gfx601", EmbDeviceCode::GFX601},
      {"gfx602", EmbDeviceCode::GFX602},
      {"gfx700", EmbDeviceCode::GFX700},
      {"gfx701", EmbDeviceCode::GFX701},
      {"gfx702", EmbDeviceCode::GFX702},
      {"gfx703", EmbDeviceCode::GFX703},
      {"gfx704", EmbDeviceCode::GFX704},
      {"gfx705", EmbDeviceCode::GFX705},
      {"gfx801", EmbDeviceCode::GFX801},
      {"gfx802", EmbDeviceCode::GFX802},
      {"gfx803", EmbDeviceCode::GFX803},
      {"gfx805", EmbDeviceCode::GFX805},
      {"gfx810", EmbDeviceCode::GFX810},
      {"gfx900", EmbDeviceCode::GFX900},
      {"gfx902", EmbDeviceCode::GFX902},
      {"gfx904", EmbDeviceCode::GFX904},
      {"gfx906", EmbDeviceCode::GFX906},
      {"gfx908", EmbDeviceCode::GFX908},
      {"gfx90a", EmbDeviceCode::GFX90A},
      {"gfx90c", EmbDeviceCode::GFX90C},
      {"gfx940", EmbDeviceCode::GFX940},
      {"gfx941", EmbDeviceCode::GFX941},
      {"gfx942", EmbDeviceCode::GFX942},
      {"gfx950", EmbDeviceCode::GFX950},
      {"gfx1010", EmbDeviceCode::GFX1010},
      {"gfx1011", EmbDeviceCode::GFX1011},
      {"gfx1012", EmbDeviceCode::GFX1012},
      {"gfx1013", EmbDeviceCode::GFX1013},
      {"gfx1030", EmbDeviceCode::GFX1030},
      {"gfx1031", EmbDeviceCode::GFX1031},
      {"gfx1032", EmbDeviceCode::GFX1032},
      {"gfx1033", EmbDeviceCode::GFX1033},
      {"gfx1034", EmbDeviceCode::GFX1034},
      {"gfx1035", EmbDeviceCode::GFX1035},
      {"gfx1036", EmbDeviceCode::GFX1036},
      {"gfx1100", EmbDeviceCode::GFX1100},
      {"gfx1101", EmbDeviceCode::GFX1101},
      {"gfx1102", EmbDeviceCode::GFX1102},
      {"gfx1103", EmbDeviceCode::GFX1103},
      {"gfx1150", EmbDeviceCode::GFX1150},
      {"gfx1151", EmbDeviceCode::GFX1151},
      {"gfx1152", EmbDeviceCode::GFX1152},
      {"gfx1153", EmbDeviceCode::GFX1153},
      {"gfx1200", EmbDeviceCode::GFX1200},
      {"gfx1201", EmbDeviceCode::GFX1201},

      // NVIDIA cuda architecture
      {"sm_30", EmbDeviceCode::SM_30},
      {"sm_32", EmbDeviceCode::SM_32},
      {"sm_35", EmbDeviceCode::SM_35},
      {"sm_37", EmbDeviceCode::SM_37},
      {"sm_50", EmbDeviceCode::SM_50},
      {"sm_52", EmbDeviceCode::SM_52},
      {"sm_53", EmbDeviceCode::SM_53},
      {"sm_60", EmbDeviceCode::SM_60},
      {"sm_61", EmbDeviceCode::SM_61},
      {"sm_62", EmbDeviceCode::SM_62},
      {"sm_70", EmbDeviceCode::SM_70},
      {"sm_72", EmbDeviceCode::SM_72},
      {"sm_75", EmbDeviceCode::SM_75},
      {"sm_80", EmbDeviceCode::SM_80},
      {"sm_86", EmbDeviceCode::SM_86},
      {"sm_87", EmbDeviceCode::SM_87},
      {"sm_89", EmbDeviceCode::SM_89},
      {"sm_90", EmbDeviceCode::SM_90},
      {"sm_90a", EmbDeviceCode::SM_90A},
      {"sm_100", EmbDeviceCode::SM_100},
      {"sm_100a", EmbDeviceCode::SM_100A},
      {"sm_101", EmbDeviceCode::SM_101},
      {"sm_101a", EmbDeviceCode::SM_101A},
      {"sm_120", EmbDeviceCode::SM_120},
      {"sm_120a", EmbDeviceCode::SM_120A},

      // NVIDIA PTX virtual architecture
      {"compute_30", EmbDeviceCode::COMPUTE_30},
      {"compute_32", EmbDeviceCode::COMPUTE_32},
      {"compute_35", EmbDeviceCode::COMPUTE_35},
      {"compute_37", EmbDeviceCode::COMPUTE_37},
      {"compute_50", EmbDeviceCode::COMPUTE_50},
      {"compute_52", EmbDeviceCode::COMPUTE_52},
      {"compute_53", EmbDeviceCode::COMPUTE_53},
      {"compute_60", EmbDeviceCode::COMPUTE_60},
      {"compute_61", EmbDeviceCode::COMPUTE_61},
      {"compute_62", EmbDeviceCode::COMPUTE_62},
      {"compute_70", EmbDeviceCode::COMPUTE_70},
      {"compute_72", EmbDeviceCode::COMPUTE_72},
      {"compute_75", EmbDeviceCode::COMPUTE_75},
      {"compute_80", EmbDeviceCode::COMPUTE_80},
      {"compute_86", EmbDeviceCode::COMPUTE_86},
      {"compute_87", EmbDeviceCode::COMPUTE_87},
      {"compute_89", EmbDeviceCode::COMPUTE_89},
      {"compute_90", EmbDeviceCode::COMPUTE_90},
      {"compute_90a", EmbDeviceCode::COMPUTE_90A},
      {"compute_100", EmbDeviceCode::COMPUTE_100},
      {"compute_100a", EmbDeviceCode::COMPUTE_100A},
      {"compute_101", EmbDeviceCode::COMPUTE_101},
      {"compute_101a", EmbDeviceCode::COMPUTE_101A},
      {"compute_120", EmbDeviceCode::COMPUTE_120},
      {"compute_120a", EmbDeviceCode::COMPUTE_120A},
  };

  if (not archs.contains(s))
    return createStringError("Cannot convert string to EmbDeviceCode::Id");
  return archs.at(s);
}

bool EmbDeviceCode::isArchive() const { return object::isArchive(code); }

bool EmbDeviceCode::isObject() const { return object::isObject(code); }

bool EmbDeviceCode::isShared() const { return object::isShared(code); }

Expected<EmbDeviceCode::Id> EmbDeviceCode::getIdFor(uint64_t n) {
  uint64_t unused = n & maskUnused;
  auto fmt = static_cast<BinaryFormat>(n & maskFormat);

  if (unused == 0) {
    Id id = Id(n);
    if (fmt == BinaryFormat::AMDGPU) {
      if (id > Id::AMDGPU_lo and id < AMDGPU_hi)
        return id;
    } else if (fmt == BinaryFormat::NVSASS) {
      if (id > Id::NVSASS_lo and id < Id::NVSASS_hi)
        return id;
    } else if (fmt == BinaryFormat::NVPTX) {
      if (id > Id::NVPTX_lo and id < Id::NVPTX_hi)
        return id;
    }
  }
  return createStringError("Cannot convert integer to EmbDeviceCode::Id");
}

// Expected<std::optional<EmbDeviceCode>>
// EmbDeviceCode::parse(const ObjectFile &objFile, TTID tt) {
//   for (SectionRef sec : objFile.sections())
//     if (isDeviceCodeSection(sec, tt))
//       return parseSection(sec);
//   return std::nullopt;
// }

// Expected<std::optional<EmbDeviceCode>>
// EmbDeviceCode::parse(const Archive &archive, TTID tt) {
//   Error err = Error::success();
//   for (const Archive::Child &child : archive.children(err)) {
//     if (err)
//       return err;

//     Expected<MemoryBufferRef> memBufOrErr = child.getMemoryBufferRef();
//     if (not memBufOrErr)
//       return memBufOrErr.takeError();
//     const MemoryBufferRef &memBuf = *memBufOrErr;

//     Expected<std::unique_ptr<ObjectFile>> objFileOrErr =
//         ObjectFile::createObjectFile(memBuf);
//     if (not objFileOrErr)
//       return objFileOrErr.takeError();
//     const ObjectFile &objFile = **objFileOrErr;

//     Expected<std::optional<EmbDeviceCode>> devCodeOrErr = parse(objFile, tt);
//     if (not devCodeOrErr)
//       return devCodeOrErr.takeError();
//     std::optional<EmbDeviceCode> devCode = *devCodeOrErr;

//     if (devCode)
//       linker.add(*devCode);
//   }

//   return linker.linkStatic();
// }

// Expected<std::optional<EmbDeviceCode>> EmbDeviceCode::create(const Binary
// &bin,
//                                                              TTID tt) {
//   if (auto *objFile = dyn_cast<ObjectFile>(&bin))
//     return EmbDeviceCode::create(*objFile, tt);
//   else if (auto *archive = dyn_cast<Archive>(&bin))
//     return EmbDeviceCode::create(*archive, tt);
//   llvm_unreachable("EmbDeviceCode::create: File format not supported");
// }
