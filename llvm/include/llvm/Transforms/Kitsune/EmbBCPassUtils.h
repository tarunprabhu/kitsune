//==- EmbBCPassUtils.h - Utilities for embedded bitcode passes --*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Utilities shared by embedded bitcode passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_KITSUNE_EMB_BC_PASS_UTILS_H
#define LLVM_TRANSFORMS_KITSUNE_EMB_BC_PASS_UTILS_H

#include "llvm/IR/Module.h"

#include <memory>

namespace llvm {

class LLVMContext;

/// Parse a libdevice bitcode file. This is expected to succeed, so any failure
/// to parse the file will result in a catastrophic error.
/// @filename Full path to the bitcode file
/// @ctx The LLVM context
std::unique_ptr<Module> parseLibDeviceBCFile(StringRef file, LLVMContext &ctx);

} // namespace llvm

#endif // LLVM_TRANSFORMS_KITSUNE_EMB_BC_PASS_UTILS_H
