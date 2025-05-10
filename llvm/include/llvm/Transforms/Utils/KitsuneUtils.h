//===- KitsuneUtils.h - Helper functions for Kitsune -----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kitsune-specific utilities. This really could go anywhere, and, arguably,
// TapirUtils.h is probably a better place for these, but most of the utility
// functions there are very specific to tapir tasks and other Tapir-specific
// constructs. Most of the utilities here are intended to help deal with
// Kitsune-specific intrinsics and tapir targets
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_KITSUNE_UTILS_H
#define LLVM_TRANSFORMS_UTILS_KITSUNE_UTILS_H

#include "llvm/Frontend/Tapir/Tapir.h"
#include "llvm/IR/Constants.h"

namespace llvm {

class LLVMContext;

/// Generate a ConstantInt for use in Kitsune-specific intrinsics that take a
/// tapir target id as an argument.
ConstantInt *getConstantInt(LLVMContext &ctxt, TapirTargetID tt);

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_KITSUNE_UTILS_H
