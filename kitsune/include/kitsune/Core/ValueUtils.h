//===- ValueUtils.h - Utilities for LLVM Value's ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM values.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_VALUE_UTILS_H
#define KITSUNE_CORE_VALUE_UTILS_H

namespace llvm {

class Type;
class Value;

/// \addtogroup kitsune
/// @{

/// Check that the given value is a constant 0 of integer or floating point
/// type.
bool isZero(const Value *v);

/// Check that the given value is a constant 0 of the given type.
bool isZero(const Value *v, Type *ty);

/// Check that the given value is a constant 1 of integer type.
bool isIntOne(const Value *v);

/// Check that the given value is a constant 1 of the given type.
bool isIntOne(const Value *v, Type *ty);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_VALUE_UTILS_H
