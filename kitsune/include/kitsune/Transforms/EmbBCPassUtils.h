//==- EmbBCPassUtils.h - Utilities for embedded bitcode passes --*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities shared by embedded bitcode passes.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_EMB_BC_PASS_UTILS_H
#define KITSUNE_TRANSFORMS_EMB_BC_PASS_UTILS_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"

#include <memory>

namespace llvm {

class LLVMContext;
class TapirTargetOptions;

/// Generate a module containing the contents of the libDevice bitcode file(s)
/// for the given tapir target. This will parse one or more files as specified
/// in the given tapir target options. Calling this function multiple times can
/// get expensive, so callers should cache this module if possible.
std::unique_ptr<Module>
getLibDeviceModule(TTID tt, const TapirTargetOptions &ttOpts, LLVMContext &ctx);

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_EMB_BC_PASS_UTILS_H
