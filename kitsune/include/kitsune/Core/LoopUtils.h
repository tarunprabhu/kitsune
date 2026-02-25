//===- LoopUtils.h - Utilities for LLVM loops ------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM loops.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_LOOP_UTILS_H
#define KITSUNE_CORE_LOOP_UTILS_H

#include "llvm/ADT/SmallVector.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

class Loop;

/// Remove all attributes specific to tapir loops from the given loop.
void clearTapirLoopAttrs(Loop &loop);

/// Recursively gather all subloops of the given loop.
SmallVector<Loop *, 4> getAllSubLoops(Loop &loop);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_LOOP_UTILS_H
