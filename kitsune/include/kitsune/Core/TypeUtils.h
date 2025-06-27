//=- EmbUtils.h - Helper functions for types ---------------------*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for types.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_TYPE_UTILS_H
#define KITSUNE_CORE_TYPE_UTILS_H

namespace llvm {

class Type;

/// Is the type an array of bytes.
bool isByteArrayTy(Type *ty);

} // namespace llvm

#endif // KITSUNE_CORE_TYPE_UTILS_H
