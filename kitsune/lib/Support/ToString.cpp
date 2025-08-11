//=- ToString.cpp - String and serialization functions ------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implementation of additional string utilities and serialization functions.
///
//===----------------------------------------------------------------------===//

#include "kitsune/Support/ToString.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

std::string llvm::toString(const TTID &tt) {
  // clang-format off
  switch (tt) {
  case TTID::Nolo: return "nolo";
  case TTID::Serial: return "serial";
  case TTID::Cuda: return "cuda";
  case TTID::Hip: return "hip";
  case TTID::Lambda: return "lambda";
  case TTID::OMPTask: return "omptask";
  case TTID::OpenCilk: return "opencilk";
  case TTID::OpenMP: return "openmp";
  case TTID::Qthreads: return "qthreads";
  case TTID::Realm: return "realm";
  }
  // clang-format off
  llvm_unreachable("toString: TTID not handled");
}

std::string llvm::toString(const MaybeBool &v) {
  // clang-format off
  switch (v) {
  case MaybeBool::Off: return "off";
  case MaybeBool::On: return "on";
  case MaybeBool::Any: return "any";
  }
  // clang-format on
  llvm_unreachable("toString: MaybeBool value not handled");
}

std::string llvm::toString(const OptznLevel &optLevel) {
  // clang-format off
  switch (optLevel) {
  case OptznLevel::O0: return "O0";
  case OptznLevel::O1: return "O1";
  case OptznLevel::O2: return "O2";
  case OptznLevel::O3: return "O3";
  case OptznLevel::Os: return "Os";
  case OptznLevel::Oz: return "Os";
  }
  // clang-format on
  llvm_unreachable("operator<<: OptznLevel not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const TTID &tt) {
  os << toString(tt);
  return os;
}

raw_ostream &llvm::operator<<(raw_ostream &os, const std::optional<TTID> &tt) {
  if (tt.has_value())
    return os << *tt;
  else
    return os << "<<<std::nullopt>>>";
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const TapirSpawnStrategy &strategy) {
  switch (strategy) {
  case TapirSpawnStrategy::Sequential:
    return os << "Sequential";
  case TapirSpawnStrategy::DivideAndConquer:
    return os << "Divide and conquer";
  case TapirSpawnStrategy::GPU:
    return os << "GPU";
  }
  llvm_unreachable("operator<<: TapirSpawnStrategy not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const MaybeBool &v) {
  os << toString(v);
  return os;
}

raw_ostream &llvm::operator<<(raw_ostream &os, const OptznLevel &optLevel) {
  os << toString(optLevel);
  return os;
}

raw_ostream &llvm::operator<<(raw_ostream &os, const CodeGenOptLevel &cgOpt) {
  // clang-format off
  switch (cgOpt) {
  case CodeGenOptLevel::None: return os << "none (O0)";
  case CodeGenOptLevel::Less: return os << "less (O1)";
  case CodeGenOptLevel::Default: return os << "default (O2)";
  case CodeGenOptLevel::Aggressive: return os << "aggressive (O3)";
  }
  // clang-format on
  llvm_unreachable("operator<<: CodeGenOptLevel not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const CodeModel::Model &cm) {
  // clang-format off
  switch (cm) {
  case CodeModel::Tiny: return os << "tiny";
  case CodeModel::Small: return os << "small";
  case CodeModel::Kernel: return os << "kernel";
  case CodeModel::Medium: return os << "medium";
  case CodeModel::Large: return os << "large";
  }
  // clang-format on
  llvm_unreachable("operator<<: CodeModel not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const Reloc::Model &rm) {
  // clang-format off
  switch (rm) {
  case Reloc::Static: return os << "static";
  case Reloc::PIC_: return os << "pic";
  case Reloc::DynamicNoPIC: return os << "dynamic-no-pic";
  case Reloc::ROPI: return os << "ropi";
  case Reloc::RWPI: return os << "rwpi";
  case Reloc::ROPI_RWPI: return os << "ropi-rwpi";
  }
  // clang-format on
  llvm_unreachable("operator<<: RelocModel not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const BasicBlockSection &sect) {
  // clang-format off
  switch (sect) {
  case BasicBlockSection::All: return os << "all";
  case BasicBlockSection::List: return os << "list";
  case BasicBlockSection::Preset: return os << "preset";
  case BasicBlockSection::None: return os << "none";
  }
  // clang-format on
  llvm_unreachable("operator<<: BasicBlockSection not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const DebuggerKind &dbgKind) {
  // clang-format off
  switch (dbgKind) {
  case DebuggerKind::Default: return os << "default";
  case DebuggerKind::GDB: return os << "gdb";
  case DebuggerKind::LLDB: return os << "lldb";
  case DebuggerKind::SCE: return os << "sce";
  case DebuggerKind::DBX: return os << "dbx";
  }
  // clang-format on
  llvm_unreachable("operator<<: DebuggerKind not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const EABI &eabi) {
  // clang-format off
  switch (eabi) {
  case EABI::Unknown: return os << "unknown";
  case EABI::Default: return os << "default";
  case EABI::EABI4: return os << "eabi4";
  case EABI::EABI5: return os << "eabi5";
  case EABI::GNU: return os << "gnu";
  }
  // clang-format on
  llvm_unreachable("operator<<: EABI not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const FloatABI::ABIType &abi) {
  // clang-format off
  switch (abi) {
  case FloatABI::Default: return os << "default";
  case FloatABI::Soft: return os << "soft";
  case FloatABI::Hard: return os << "hard";
  }
  // clang-format on
  llvm_unreachable("operator<<: FloatABI::ABIType not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const FPOpFusion::FPOpFusionMode &mode) {
  // clang-format off
  switch (mode) {
  case FPOpFusion::Fast: return os << "fast";
  case FPOpFusion::Standard: return os << "standard";
  case FPOpFusion::Strict: return os << "strict";
  }
  // clang-format on
  llvm_unreachable("operator<<: FPOpFusionMode not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const GlobalISelAbortMode &m) {
  switch (m) {
  case GlobalISelAbortMode::Disable:
    return os << "disable";
  case GlobalISelAbortMode::Enable:
    return os << "enable";
  case GlobalISelAbortMode::DisableWithDiag:
    return os << "disable-with-diag";
  }
  llvm_unreachable("operator<<: GlobalISelAbortMode not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const SwiftAsyncFramePointerMode &mode) {
  switch (mode) {
  case SwiftAsyncFramePointerMode::DeploymentBased:
    return os << "deployment-based";
  case SwiftAsyncFramePointerMode::Always:
    return os << "always";
  case SwiftAsyncFramePointerMode::Never:
    return os << "never";
  }
  llvm_unreachable("operator<<: SwiftAsyncFramePointerMode not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const ThreadModel::Model &model) {
  // clang-format off
  switch (model) {
  case ThreadModel::POSIX: return os << "posix";
  case ThreadModel::Single: return os << "single";
  }
  // clang-format on
  llvm_unreachable("operator<<: ThreadModel::Model not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const MCTargetOptions::DwarfDirectory &dd) {
  switch (dd) {
  case MCTargetOptions::DisableDwarfDirectory:
    return os << "disable";
  case MCTargetOptions::EnableDwarfDirectory:
    return os << "enable";
  case MCTargetOptions::DefaultDwarfDirectory:
    return os << "default";
  }
  llvm_unreachable("operator<<: DwarfDirectory not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const DebugCompressionType &dc) {
  // clang-format off
  switch (dc) {
  case DebugCompressionType::None: return os << "none";
  case DebugCompressionType::Zlib: return os << "zlib";
  case DebugCompressionType::Zstd: return os << "zstd";
  }
  // clang-format on
  llvm_unreachable("operator<<: DebugCompressionType not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const EmitDwarfUnwindType &du) {
  switch (du) {
  case EmitDwarfUnwindType::Always:
    return os << "always";
  case EmitDwarfUnwindType::NoCompactUnwind:
    return os << "no-compact-unwind";
  case EmitDwarfUnwindType::Default:
    return os << "default";
  }
  llvm_unreachable("operator<<: EmitDwarfUnwindType not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const std::error_code &ec) {
  return os << ec.message();
}

raw_ostream &llvm::operator<<(raw_ostream &os, const file_magic::Impl &magic) {
  switch (magic) {
  case file_magic::unknown:
    return os << "unknown";
  case file_magic::bitcode:
    return os << "bitcode";
  case file_magic::clang_ast:
    return os << "clang-ast";
  case file_magic::archive:
    return os << "archive";
  case file_magic::elf:
    return os << "elf";
  case file_magic::elf_relocatable:
    return os << "elf-relocatable";
  case file_magic::elf_executable:
    return os << "elf-executable";
  case file_magic::elf_shared_object:
    return os << "elf-shared-object";
  case file_magic::elf_core:
    return os << "elf-core";
  case file_magic::goff_object:
    return os << "goff-object";
  case file_magic::macho_object:
    return os << "macho-object";
  case file_magic::macho_executable:
    return os << "macho-executable";
  case file_magic::macho_fixed_virtual_memory_shared_lib:
    return os << "macho-fixed-virtual-memory-shared-lib";
  case file_magic::macho_core:
    return os << "macho";
  case file_magic::macho_preload_executable:
    return os << "macho-preload-executable";
  case file_magic::macho_dynamically_linked_shared_lib:
    return os << "macho-dynamically-linked-shared-lib";
  case file_magic::macho_dynamic_linker:
    return os << "macho-dynamic-linker";
  case file_magic::macho_bundle:
    return os << "macho-bundle";
  case file_magic::macho_dynamically_linked_shared_lib_stub:
    return os << "macho-dynamically-linked-shared-lib-stub";
  case file_magic::macho_dsym_companion:
    return os << "macho-dsym-companion";
  case file_magic::macho_kext_bundle:
    return os << "macho-kext-bundle";
  case file_magic::macho_universal_binary:
    return os << "macho-universal-binary";
  case file_magic::macho_file_set:
    return os << "macho-file-set";
  case file_magic::minidump:
    return os << "minidump";
  case file_magic::coff_cl_gl_object:
    return os << "coff-cl-gl-object";
  case file_magic::coff_object:
    return os << "coff-object";
  case file_magic::coff_import_library:
    return os << "coff-import-library";
  case file_magic::pecoff_executable:
    return os << "pecoff-executable";
  case file_magic::windows_resource:
    return os << "windows-resource";
  case file_magic::xcoff_object_32:
    return os << "xcoff-object-32";
  case file_magic::xcoff_object_64:
    return os << "xcoff-object-64";
  case file_magic::wasm_object:
    return os << "wasm-object";
  case file_magic::pdb:
    return os << "pdb";
  case file_magic::tapi_file:
    return os << "tapi";
  case file_magic::cuda_fatbinary:
    return os << "cuda-fatbinary";
  case file_magic::offload_binary:
    return os << "offload-binary";
  case file_magic::dxcontainer_object:
    return os << "dxcontainer-object";
  case file_magic::offload_bundle:
    return os << "offload-bundle";
  case file_magic::offload_bundle_compressed:
    return os << "offload-bundle-compressed";
  case file_magic::spirv_object:
    return os << "spirv-object";
  }
  llvm_unreachable("EmbDeviceCodeParser::toString: file magic not handled");
}
