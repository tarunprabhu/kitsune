//===- KitsuneUtils.h - Kitsune-specific utilities -------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for the Kitsune-specific tapir targets. Moving these to a Kitsune
// support library is more trouble than it is worth.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_KITSUNE_UTILS_H
#define LLVM_TRANSFORMS_TAPIR_KITSUNE_UTILS_H

#include "llvm/ADT/StringRef.h"

#include <set>

namespace llvm {

class BasicBlock;
class Function;
class GlobalValue;
class Loop;
class Module;
class TapirLoopInfo;

/// Collect the GlobalValues used in a \ref BasicBlock.
/// @param bb The basic block
/// @param seen The set into which to collect the GlobalValue's
void collectGlobalValues(llvm::BasicBlock &bb,
                         std::set<llvm::GlobalValue *> &seen);

/// Collect the GlobalValues used in a \ref Function.
/// @param f The function
/// @param seen The set into which to collect the GlobalValue's
void collectGlobalValues(Function &f, std::set<GlobalValue *> &seen);

/// Get the GlobalValues used in a \ref Loop.
/// @param loop The loop
/// @param seen The set into which to collect the GlobalValue's
void collectGlobalValues(Loop &loop, std::set<GlobalValue *> &seen);

/// Construct the name to be used for the outlined function consisting of the
/// body of the tapir loop.
std::string getNameForTapirLoop(const TapirLoopInfo &tl, StringRef prefix = "");

/// Construct the name for a device module.
std::string getNameForDeviceModule(const Module &hostM, StringRef prefix = "");

} // namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_KITSUNE_UTILS_H
