//===- TapirGPUUtils.h - Helpers for GPU-centric tapir targets -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Utility functions shared by the GPU-centric tapir targets
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TAPIR_GPU_UTILS_H
#define LLVM_TAPIR_GPU_UTILS_H

#include <set>

// FIXME: Move everything in this file into a different kitsune utilities
// file.

namespace llvm {

class BasicBlock;
class Function;
class GlobalValue;
class Loop;

/// Collect the GlobalValues used in a basic block. This includes Functions
/// and GlobalVariables, but also GlobalAliases and GlobalIFunc's.
/// @param bb The basic block
/// @param seen The set into which to collect the GlobalValue's
void collectGlobalValues(llvm::BasicBlock &bb,
                         std::set<llvm::GlobalValue *> &seen);

/// Collect the GlobalValues used in a Function. This includes Functions and
/// GlobalVariables, but also GlobalAliases and GlobalIFunc's.
/// @param f The function
/// @param seen The set into which to collect the GlobalValue's
void collectGlobalValues(Function &f, std::set<GlobalValue *> &seen);

/// Get the global values used in a loop. This includes Functions and
/// GlobalVariables, but also GlobalAliases and GlobalIFunc's.
/// @param loop The loop
/// @param seen The set into which to collect the GlobalValue's
void collectGlobalValues(Loop &loop, std::set<GlobalValue *> &seen);

} // namespace llvm

#endif // LLVM_TAPIR_GPU_UTILS_H
