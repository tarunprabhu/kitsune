//===- OstreamUtils.h - Utilities to stream types --------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to stream certain types. These are primarily used to stream enums
// which are then rendered as text. These are mainly useful for debugging.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_OSTREAM_UTILS_H
#define KITSUNE_SUPPORT_OSTREAM_UTILS_H

#include "kitsune/Core/OptznLevel.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetOptions.h"

#include <optional>

namespace llvm {

/// \addtogroup kitsune
/// @{

class raw_ostream;

/// @{
/// Helpers to stream Kitsune-specific types.

raw_ostream &operator<<(raw_ostream &os, const std::optional<TTID> &v);
raw_ostream &operator<<(raw_ostream &os, const TTID &v);
raw_ostream &operator<<(raw_ostream &os, const TapirSpawnStrategy &v);
raw_ostream &operator<<(raw_ostream &os, const OptznLevel &v);
raw_ostream &operator<<(raw_ostream &os, const MaybeBool &v);

/// @}

/// @{
/// Helpers to stream types used in CodeGen.

raw_ostream &operator<<(raw_ostream &os, const CodeGenOptLevel &v);
raw_ostream &operator<<(raw_ostream &os, const CodeModel::Model &v);
raw_ostream &operator<<(raw_ostream &os, const Reloc::Model &v);

/// @}

/// @{
/// Helpers to stream types used in TargetOptions.

raw_ostream &operator<<(raw_ostream &os, const BasicBlockSection &v);
raw_ostream &operator<<(raw_ostream &os, const DebuggerKind &v);
raw_ostream &operator<<(raw_ostream &os, const EABI &v);
raw_ostream &operator<<(raw_ostream &os, const FloatABI::ABIType &v);
raw_ostream &operator<<(raw_ostream &os, const FPOpFusion::FPOpFusionMode &v);
raw_ostream &operator<<(raw_ostream &os, const GlobalISelAbortMode &v);
raw_ostream &operator<<(raw_ostream &os, const SwiftAsyncFramePointerMode &v);
raw_ostream &operator<<(raw_ostream &os, const ThreadModel::Model &v);

/// @}

/// @{
/// Helpers to stream types used in MCTargetOptions.

raw_ostream &operator<<(raw_ostream &os, const DebugCompressionType &v);
raw_ostream &operator<<(raw_ostream &os,
                        const MCTargetOptions::DwarfDirectory &v);
raw_ostream &operator<<(raw_ostream &os, const EmitDwarfUnwindType &v);

/// @}

/// @}

} // namespace llvm

#endif // KITSUNE_SUPPORT_OSTREAM_UTILS_H
