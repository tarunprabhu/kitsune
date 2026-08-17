//===- ToString.cpp - String and serialization functions ------------------===//
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
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

template <> StringRef llvm::toString<int8_t>() { return "int8_t"; }
template <> StringRef llvm::toString<uint8_t>() { return "uint8_t"; }
template <> StringRef llvm::toString<int16_t>() { return "int16_t"; }
template <> StringRef llvm::toString<uint16_t>() { return "uint16_t"; }
template <> StringRef llvm::toString<int32_t>() { return "int32_t"; }
template <> StringRef llvm::toString<uint32_t>() { return "uint32_t"; }
template <> StringRef llvm::toString<int64_t>() { return "int64_t"; }
template <> StringRef llvm::toString<uint64_t>() { return "uint64_t"; }
template <> StringRef llvm::toString<float>() { return "float"; }
template <> StringRef llvm::toString<double>() { return "double"; }

template <> std::string llvm::toString(const bool &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const int8_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const uint8_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const int16_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const uint16_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const int32_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const uint32_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const int64_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const uint64_t &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const float &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const double &v) {
  return std::to_string(v);
}

template <> std::string llvm::toString(const char *s) { return s; }

template <> std::string llvm::toString(const std::string &s) { return s; }

template <> std::string llvm::toString(const StringRef &s) { return s.str(); }

template <> std::string llvm::toString(const CodeGenOptLevel &cgOpt) {
  switch (cgOpt) {
  case CodeGenOptLevel::None: return "none (O0)";
  case CodeGenOptLevel::Less: return "less (O1)";
  case CodeGenOptLevel::Default: return "default (O2)";
  case CodeGenOptLevel::Aggressive: return "aggressive (O3)";
  }
  llvm_unreachable("operator<<: CodeGenOptLevel not handled");
}

template <> std::string llvm::toString(const CodeModel::Model &cm) {
  switch (cm) {
  case CodeModel::Tiny: return "tiny";
  case CodeModel::Small: return "small";
  case CodeModel::Kernel: return "kernel";
  case CodeModel::Medium: return "medium";
  case CodeModel::Large: return "large";
  }
  llvm_unreachable("operator<<: CodeModel not handled");
}

template <> std::string llvm::toString(const Reloc::Model &rm) {
  switch (rm) {
  case Reloc::Static: return "static";
  case Reloc::PIC_: return "pic";
  case Reloc::DynamicNoPIC: return "dynamic-no-pic";
  case Reloc::ROPI: return "ropi";
  case Reloc::RWPI: return "rwpi";
  case Reloc::ROPI_RWPI: return "ropi-rwpi";
  }
  llvm_unreachable("operator<<: RelocModel not handled");
}

template <> std::string llvm::toString(const BasicBlockSection &sect) {
  switch (sect) {
  case BasicBlockSection::All: return "all";
  case BasicBlockSection::List: return "list";
  case BasicBlockSection::Preset: return "preset";
  case BasicBlockSection::None: return "none";
  }
  llvm_unreachable("operator<<: BasicBlockSection not handled");
}

template <> std::string llvm::toString(const DebuggerKind &dbgKind) {
  switch (dbgKind) {
  case DebuggerKind::Default: return "default";
  case DebuggerKind::GDB: return "gdb";
  case DebuggerKind::LLDB: return "lldb";
  case DebuggerKind::SCE: return "sce";
  case DebuggerKind::DBX: return "dbx";
  }
  llvm_unreachable("operator<<: DebuggerKind not handled");
}

template <> std::string llvm::toString(const EABI &eabi) {
  switch (eabi) {
  case EABI::Unknown: return "unknown";
  case EABI::Default: return "default";
  case EABI::EABI4: return "eabi4";
  case EABI::EABI5: return "eabi5";
  case EABI::GNU: return "gnu";
  }
  llvm_unreachable("operator<<: EABI not handled");
}

template <> std::string llvm::toString(const FloatABI::ABIType &abi) {
  switch (abi) {
  case FloatABI::Default: return "default";
  case FloatABI::Soft: return "soft";
  case FloatABI::Hard: return "hard";
  }
  llvm_unreachable("operator<<: FloatABI::ABIType not handled");
}

template <> std::string llvm::toString(const FPOpFusion::FPOpFusionMode &mode) {
  switch (mode) {
  case FPOpFusion::Fast: return "fast";
  case FPOpFusion::Standard: return "standard";
  case FPOpFusion::Strict: return "strict";
  }
  llvm_unreachable("operator<<: FPOpFusionMode not handled");
}

template <> std::string llvm::toString(const GlobalISelAbortMode &m) {
  switch (m) {
  case GlobalISelAbortMode::Disable: return "disable";
  case GlobalISelAbortMode::Enable: return "enable";
  case GlobalISelAbortMode::DisableWithDiag: return "disable-with-diag";
  }
  llvm_unreachable("operator<<: GlobalISelAbortMode not handled");
}

template <> std::string llvm::toString(const SwiftAsyncFramePointerMode &mode) {
  switch (mode) {
  case SwiftAsyncFramePointerMode::DeploymentBased: return "deployment-based";
  case SwiftAsyncFramePointerMode::Always: return "always";
  case SwiftAsyncFramePointerMode::Never: return "never";
  }
  llvm_unreachable("operator<<: SwiftAsyncFramePointerMode not handled");
}

template <> std::string llvm::toString(const ThreadModel::Model &model) {
  switch (model) {
  case ThreadModel::POSIX: return "posix";
  case ThreadModel::Single: return "single";
  }
  llvm_unreachable("operator<<: ThreadModel::Model not handled");
}

template <>
std::string llvm::toString(const MCTargetOptions::DwarfDirectory &dd) {
  switch (dd) {
  case MCTargetOptions::DisableDwarfDirectory: return "disable";
  case MCTargetOptions::EnableDwarfDirectory: return "enable";
  case MCTargetOptions::DefaultDwarfDirectory: return "default";
  }
  llvm_unreachable("operator<<: DwarfDirectory not handled");
}

template <> std::string llvm::toString(const DebugCompressionType &dc) {
  switch (dc) {
  case DebugCompressionType::None: return "none";
  case DebugCompressionType::Zlib: return "zlib";
  case DebugCompressionType::Zstd: return "zstd";
  }
  llvm_unreachable("operator<<: DebugCompressionType not handled");
}

template <> std::string llvm::toString(const EmitDwarfUnwindType &du) {
  switch (du) {
  case EmitDwarfUnwindType::Always: return "always";
  case EmitDwarfUnwindType::NoCompactUnwind: return "no-compact-unwind";
  case EmitDwarfUnwindType::Default: return "default";
  }
  llvm_unreachable("operator<<: EmitDwarfUnwindType not handled");
}
