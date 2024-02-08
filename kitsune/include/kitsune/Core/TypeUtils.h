//===- TypeUtils.h - Helper functions for LLVM types -----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM types.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_TYPE_UTILS_H
#define KITSUNE_CORE_TYPE_UTILS_H

#include "llvm/IR/DerivedTypes.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

/// Is the type an array of bytes.
bool isByteArrayTy(Type *ty);

/// Get the LLVM type for a given C++ type. This only works for primitive and
/// pointer types. This cannot be used for struct/class types. However, pointers
/// to struct's or classes are allowed.
template <typename T, std::enable_if_t<!std::is_pointer_v<T>, int> = 0>
Type *getLLVMTypeFor(LLVMContext &ctx);

template <typename T, std::enable_if_t<std::is_pointer_v<T>, int> = 0>
Type *getLLVMTypeFor(LLVMContext &ctx) {
  return PointerType::getUnqual(ctx);
}

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_TYPE_UTILS_H
