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

#include "kitsune/Common/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <set>

namespace llvm {

class BasicBlock;
class Constant;
class Function;
class GlobalValue;
class Loop;
class Module;
class Type;
class raw_ostream;

/// Render a command line to the given output stream. This will typically be for
/// a subcommand run by the GPU tapir targets - typically this will be one that
/// launches ptxas, lld etc.
llvm::raw_ostream &renderCommandLine(llvm::ArrayRef<llvm::StringRef> Args,
                                     llvm::raw_ostream &os);

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

/// Create a global variable that is intended to eventually become the fat
/// binary. This will create a variable with internal linkage and an inital
/// value.
/// FIXME: This should be removed once HipABI is refactored to use the separate
/// pass strategy that is under development.
llvm::Constant *getOrInsertFBGlobal(llvm::Module &m, llvm::StringRef name,
                                    llvm::Type *ty);

/// Create a string literal.
/// FIXME: This should be removed in favor of using CreateGlobalString from
/// the IR builder.
llvm::Constant *createConstantStr(const std::string &str, llvm::Module &m,
                                  const std::string &name = "",
                                  const std::string &section = "",
                                  unsigned align = 0);

// FIXME: This should be removed in favor of the function provided in
// ModuleUtils. This can be done after the HipABI code has been refactored to
// use the separate passes that is being worked on currently.
void appendToGlobalCtors(llvm::Module &m, llvm::Constant *c, int priority,
                         llvm::Constant *data);

// FIXME: This should be removed from here when HipABI is refactored. This will
// only be present in FinalizeKernelMetadataPass.
/// Determine the counts of various instruction kinds in the given LLVM
/// function.
KernelInstMixData getKernelInstMix(const Function &f);

} // namespace llvm

#endif // LLVM_TAPIR_GPU_UTILS_H
