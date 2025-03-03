//
//===- TapirGPUUtils.h - Helpers for GPU targets ---------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TAPIR_GPU_UTILS_H
#define LLVM_TAPIR_GPU_UTILS_H

#include "llvm/IR/Constant.h"
#include "llvm/IR/Module.h"

#include <set>

namespace llvm {

namespace tapir {

/// Render a command line to stderr. This will typically be for a subcommand run
/// by the GPU tapir targets - typically this will be one that launches ptxas,
/// lld etc.
void printCommandLine(llvm::ArrayRef<llvm::StringRef> Args);

/// Collect the global values used in a basic block. This includes Functions
/// and GlobalVariables, but also GlobalAliases and GlobalIFunc's.
void collectUsedGlobalValues(llvm::BasicBlock &bb,
                             std::set<llvm::GlobalValue *> &seen);

/// Create a global variable that is intended to eventually become the fat
/// binary. This will create a variable with internal linkage and an inital
/// value.
llvm::Constant *getOrInsertFBGlobal(llvm::Module &M, llvm::StringRef Name,
                                    llvm::Type *Ty);

/// Create a string literal.
llvm::Constant *createConstantStr(const std::string &Str, llvm::Module &M,
                                  const std::string &Name = "",
                                  const std::string &SectionName = "",
                                  unsigned Alignment = 0);

void appendToGlobalCtors(llvm::Module &M, llvm::Constant *C, int Priority,
                         llvm::Constant *Data);

// NOTE: This needs to be kept up to date with the structure in the kitsune
// runtime. We currently have avoided including files between the two but
// perhaps we should...  ????
struct KernelInstMixData {
  uint64_t numMemoryOps;
  uint64_t numFlops;
  uint64_t numIntOps;
  uint64_t numOtherOps;
};

void getKernelInstructionMix(const llvm::Function *F,
                             KernelInstMixData &InstMix);

} // namespace tapir

} // namespace llvm

#endif // LLVM_TAPIR_GPU_UTILS_H
