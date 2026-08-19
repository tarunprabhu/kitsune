//===- Reductions.h - Base types and utilities for reductions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base types and utilities for Kitsune's reduction support.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_REDUCTIONS_H
#define KITSUNE_CORE_REDUCTIONS_H

#include "kitsune/Support/FromInt.h"
#include "kitsune/Support/ToString.h"

#include <cstdint>

namespace llvm {

class Constant;
class Function;
class Module;
class Type;

/// The supported builtin reduction operators. We have signed, unsigned, and
/// floating point variants of these operators in order to match the builtin
/// reduction operators supported by LLVM's AtomicRMW instruction. Those, in
/// turn,  are a superset of those provided by NVIDIA and AMD GPU's. We provide
/// support for a few additional reduction operations, in addition to fully
/// custom reductions that are unlikely to have native hardware support.
///
/// NOTE: Do not change the numerical values of these enumerations. While they
/// are largely arbitrary, many tests explicitly look for them. If they are
/// changed, a large number of tests may have to updated, and this is not at all
/// worth it.
///
/// Some numerical values are deliberately missing. This has been done to
/// accomodate "related" operators that we may support in the future. For
/// instance, the bitwise operators have values 1, 2, and 3, but 4 is currently
/// unused. This is to accomodate a NAND reduction that we may support in the
/// future. A bit of grouping is nice, and it doesn't cost us anything to leave
/// some empty slots.
enum class ReduceOp : uint32_t {
  Custom = 0,       ///< Custom (requires separate reducer function)
  And = 1,          ///< Bitwise AND
  Or = 2,           ///< Bitwise OR
  Xor = 3,          ///< Bitwise XOR
  Add = 5,          ///< Integer addition
  FAdd = 6,         ///< Floating point addition
  Mul = 7,          ///< Integer multiplication
  FMul = 8,         ///< Floating point multiplication
  FMax = 16,        ///< Equivalent to the llvm.maxnum intrinsic with nsz flag
  FMaximum = 17,    ///< Equivalent to the llvm.maximum intrinsic
  FMaximumNum = 18, ///< Equivalent to the llvm.maximumnum intrinsic
  FMin = 20,        ///< Equivalent to the llvm.minnum intrinsic with nsz flag
  FMinimum = 21,    ///< Equivalent to the llvm.minimum intrinsic
  FMinimumNum = 22, ///< Equivalent to the llvm.minimumnum intrinsic
  SMax = 24,        ///< Maximum of two signed integers
  SMin = 25,        ///< Minimum of two signed integers
  UMax = 26,        ///< Maximum of two unsigned integers
  UMin = 27,        ///< Minimum of two unsigned integers
};

/// Get the unit value with type \p ty for a reduction operator \p op.
Constant *getUnitValue(ReduceOp op, Type *ty);

/// Generate a reducer function in module \p m for the operator \p op. The
/// values that it reduces are of type \p ty.
Function *genReducer(ReduceOp op, Type *ty, Module &m);

} // namespace llvm

#endif // KITSUNE_CORE_REDUCTIONS_H
