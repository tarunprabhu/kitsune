//==- KitsuneStringExtras.h - String and serialization functions -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Additional functions that are useful when dealing with strings. Also
/// includes utilities that stringify Kitsune-specific types (though not
/// exclusively Kitsune-specific types).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_KITSUNE_STRING_EXTRAS_H
#define LLVM_SUPPORT_KITSUNE_STRING_EXTRAS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/Tapir/Tapir.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Target/TargetOptions.h"

#include <optional>

namespace llvm {

class raw_ostream;

/// @{
/// Serialization functions for various types.

// Defining this here is lousy workaround since this should really be somewhere
// in "core". But there isn't a lot of precedent for this, so we put this here
// for now until we sort out where to put these.
std::string toString(const TapirTargetID &tt);
std::string toString(const MaybeBool &);

raw_ostream &operator<<(raw_ostream &os, const std::optional<TapirTargetID> &);
raw_ostream &operator<<(raw_ostream &os, const TapirTargetID &);
raw_ostream &operator<<(raw_ostream &os, const TapirSpawnStrategy &);
raw_ostream &operator<<(raw_ostream &os, const OptimizationLevel &);
raw_ostream &operator<<(raw_ostream &os, const FPOpFusion::FPOpFusionMode &);
raw_ostream &operator<<(raw_ostream &os, const MaybeBool &);
/// @}

} // namespace llvm

#endif // LLVM_SUPPORT_KITSUNE_STRING_EXTRAS_H
