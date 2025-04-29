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
#include "llvm/Support/raw_ostream.h"

#include <set>

namespace llvm {

namespace tapir {

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

/// Create a global variable that is intended to eventually become the fat
/// binary. This will create a variable with internal linkage and an inital
/// value.
llvm::Constant *getOrInsertFBGlobal(llvm::Module &m, llvm::StringRef name,
                                    llvm::Type *ty);

/// Create a string literal.
llvm::Constant *createConstantStr(const std::string &str, llvm::Module &m,
                                  const std::string &name = "",
                                  const std::string &section = "",
                                  unsigned align = 0);

void appendToGlobalCtors(llvm::Module &m, llvm::Constant *c, int priority,
                         llvm::Constant *data);

// NOTE: This needs to be kept up to date with the structure in the kitsune
// runtime. We currently have avoided including files between the two but
// perhaps we should...  ????
struct KernelInstMixData {
  uint64_t numMemoryOps = 0;
  uint64_t numFlops = 0;
  uint64_t numIntOps = 0;
  uint64_t numOtherOps = 0;
};

/// Populate the given structure with the counts of various instruction types
/// seen in the given LLVM function.
KernelInstMixData getKernelInstructionMix(const llvm::Function &f);

} // namespace tapir

} // namespace llvm

#endif // LLVM_TAPIR_GPU_UTILS_H
