//===- TapirUtils.h - Utility methods for Tapir ----------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions to render enums and types to make debugging easier and
// verbose mode for the tapir targets more helpful.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TAPIR_DEBUG_UTILS_H
#define LLVM_TAPIR_DEBUG_UTILS_H

#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

namespace llvm {

/// Serialization functions to help in debugging/verbose mode.
raw_ostream &operator<<(raw_ostream &os, const bool &);
raw_ostream &operator<<(raw_ostream &os, const OptimizationLevel &);
raw_ostream &operator<<(raw_ostream &os, const FPOpFusion::FPOpFusionMode &);

raw_ostream &operator<<(raw_ostream &os,
                        const std::optional<TapirTargetID> &tt);

} // namespace llvm

#endif // LLVM_TAPIR_DEBUG_UTILS_H
