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

#include "kitsune/Core/TTID.h"
#include "kitsune/Support/FromInt.h"
#include "kitsune/Support/ToString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"

#include <cstdint>
#include <optional>

namespace llvm {

class Constant;
class Function;
class Loop;
class Module;
class Value;
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

/// Information about a reduction obtained by parsing a call to Kitsune's reduce
/// intrinsic. This is tightly coupled with the call itself. If the call object
/// is erased, this object will no longer be valid.
class ReductionInfo {
public:
  CallInst *call = nullptr; ///< The call to a reduce intrinsic
  TTID tt;                  ///< The TTID of the tapir reduction loop
  ReduceOp reduceOp;        ///< The reduction operator
  unsigned elemSize = 0;    ///< The size (in bytes) of the reduced result

public:
  ReductionInfo(CallInst *call);

  /// Get the operand corresponding to the tapir target in the call.
  Value *getTTV() const { return call->getArgOperand(0); }

  /// Get the operand corresponding to the reduce operator in the call.
  Value *getReduceOpV() const { return call->getArgOperand(1); }

  /// Get the destination of the reduction.
  Value *getDest() const { return call->getArgOperand(2); }

  /// Get the operand corresponding to the element size in the call.
  Value *getElemSizeV() const { return call->getArgOperand(3); }

  /// Get the value being reduced in the call
  Value *getValue() const { return call->getArgOperand(4); }

  /// Get the unit value for this reduction.
  Value *getUnit() const { return call->getArgOperand(5); }

  /// Get the reducer for this reduction.
  Value *getReducer() const { return call->getArgOperand(6); }

  /// Get the type of the value being reduced. This simply returns the type of
  /// the value being reduced. If we ever support pointer operands here, the
  /// result of this function will not be correct since we will almost
  /// certainly be reducing the pointee in such cases.
  Type *getType() const { return getValue()->getType(); }

  /// Get a type into which the result of the reduction can be stored. This will
  /// be the same as the type of the value being reduced if the value is not a
  /// pointer. If it is a pointer, an array of bytes of suitable size will be
  /// returned.
  Type *getResultBufferType() const;

  /// Get the overload types needed by this intrinsic. This is useful when
  /// creating an equivalent call to this intrinsic.
  SmallVector<Type *, 2> getOverloadTypes() const;

  /// Get the extra arguments that are to be passed to the reducer function.
  SmallVector<Value *, 0> getExtraArgs() const;

  /// Get the type of the reducer function. This is inferred from the types of
  /// the value and any extra arguments that might be provided.
  FunctionType *getReducerType() const;

  /// Get all arguments that will be passed to a call to the reducer.
  SmallVector<Value *, 2> getReducerArgs() const;
};

/// Collect the reductions in loop \p loop.
SmallVector<ReductionInfo, 1> collectReductions(Loop &loop);

/// Get an AtomicRMWInst::BinOp corresponding to a reduction operator, if one
/// exists. Otherwise, return std::nullopt.
std::optional<AtomicRMWInst::BinOp> getAtomicOp(ReduceOp op);

/// Get the unit value with type \p ty for a reduction operator \p op.
Constant *getUnitValue(ReduceOp op, Type *ty);

/// Generate a reducer function in module \p m for the operator \p op. The
/// values that it reduces are of type \p ty.
Function *genReducer(ReduceOp op, Type *ty, Module &m);

} // namespace llvm

#endif // KITSUNE_CORE_REDUCTIONS_H
