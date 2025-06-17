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

#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/Tapir/Tapir.h"
#include "llvm/IR/Module.h"

#include <memory>

namespace llvm {

class LLVMContext;
class NamedMDNode;
class TapirTargetOptions;

/// Generate a module containing the contents of the libDevice bitcode file(s)
/// for the given tapir target. This will parse one or more files as specified
/// in the given tapir target options. Calling this function multiple times can
/// get expensive, so callers should cache this module if possible.
std::unique_ptr<Module> getLibDeviceModule(TapirTargetID tt,
                                           const TapirTargetOptions &ttOpts,
                                           LLVMContext &ctx);

} // namespace llvm

#endif // LLVM_TRANSFORMS_KITSUNE_EMB_BC_PASS_UTILS_H
