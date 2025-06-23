//==- ToString.h - String and serialization functions ------------*- C++ -*-==//
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

#ifndef KITSUNE_SUPPORT_TO_STRING_H
#define KITSUNE_SUPPORT_TO_STRING_H

#include "kitsune/Core/OptznLevel.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Target/TargetOptions.h"

#include <optional>

namespace llvm {

class raw_ostream;

/// @{
/// Serialization functions for various types.

std::string toString(const TTID &tt);
std::string toString(const MaybeBool &);

raw_ostream &operator<<(raw_ostream &os, const std::optional<TTID> &);
raw_ostream &operator<<(raw_ostream &os, const TTID &);
raw_ostream &operator<<(raw_ostream &os, const TapirSpawnStrategy &);
raw_ostream &operator<<(raw_ostream &os, const OptznLevel &);
raw_ostream &operator<<(raw_ostream &os, const FPOpFusion::FPOpFusionMode &);
raw_ostream &operator<<(raw_ostream &os, const MaybeBool &);
/// @}

} // namespace llvm

#endif // KITSUNE_SUPPORT_TO_STRING_H
