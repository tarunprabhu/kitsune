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

// Helper function for @ref concat. This is the base case. We don't bother with
// the zero-argument case because concat() is not legal. This should never be
// called directly.
template <typename Arg> void concatImpl(raw_ostream &os, Arg &&arg) {
  os << arg;
}

// Helper function for @ref concat. This should not be called directly.
template <typename Arg, typename... Args>
void concatImpl(raw_ostream &os, Arg &&arg, Args &&...args) {
  os << arg;
  concatImpl(os, args...);
}

/// Serialize the given arguments to a string and return the concatentation of
/// these string representations. The arguments must be serializable using an
/// llvm::raw_ostream. This is only really intended to construct names using a
/// combination of std::string, llvm::StringRef and integers. In principle, it
/// could be used for other things too.
template <typename Arg, typename... Args>
std::string concat(Arg &&arg, Args &&...args) {
  std::string buf;
  raw_string_ostream os(buf);

  concatImpl(os, arg, args...);
  return os.str();
}

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
