//===- Reductionutils.h - Utilities for reduction builtins ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for reduction support in frontends.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_FRONTEND_REDUCTION_UTILS_H
#define KITSUNE_FRONTEND_REDUCTION_UTILS_H

#include "kitsune/Support/FromInt.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class Constant;
class Type;

/// The supported builtin reduction operators.
///
/// NOTE: The values are explicitly enumerated because they must be kept in sync
/// with the corresponding KIT_* definitions in kitsune.h - though new operators
/// are unlikely to be added.
///
/// NOTE: It is unlikely that we will add any other operators here.
enum class ReduceOp : uint32_t {
  Custom = 0,  ///< Custom (requires separate reducer function)
  BAnd = 1,    ///< Bitwise AND
  BOr = 2,     ///< Bitwise OR
  BXor = 3,    ///< Bitwise XOR
  LAnd = 4,    ///< Logical AND
  LOr = 5,     ///< Logical OR
  LXor = 6,    ///< Logical XOR
  Max = 7,     ///< dest = std::max(dest, v)
  MaxLoc = 8,  ///< Index of the maximum value
  Min = 9,     ///< dest = std::min(dest, v)
  MinLoc = 10, /// Index of the minimum value
  Prod = 11,   ///< Multiplication
  Sum = 12,    ///< Addition
};

/// Convert a reduction operator to a string. This is mostly useful for
/// diagnostic messages.
StringRef toString(ReduceOp op);

/// Get the unit value for a reduction operator \p op with type \p ty. This
/// is generally used when \p ty is an integral type.
Constant *getUnitValueFor(ReduceOp op, Type *ty, bool isSigned);

/// Get the unit value for a reduction operator \p op with type \p ty. This
/// should be used when \p ty is a floating point, or boolean type i.e. with
/// types where there are no signed and unsigned variants. This can also be used
/// with an integral type, in which case, `isSigned` is assumed to be false.
Constant *getUnitValueFor(ReduceOp op, Type *ty);

} // namespace llvm

#endif // KITSUNE_FRONTEND_REDUCTION_UTILS_H
