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

/// The various reduction strategies that are available. This is the strategy
/// used to perform the final reduction into the result.
enum ReductionStrategy {
  /// Let the compiler pick a suitable strategy. This should only be used by the
  /// frontend when first generating an intrinsic. By the time the intrinsic is
  /// to be lowered, any uses of this should have been replaced with a specific
  /// strategy.
  Auto = 0,

  /// This is the equivalent of a serial reduction. It is performed directly
  /// into the destination without an atomic. This should only be used when a
  /// serial reduction is to be performed. In practice, with Kitsune, this will
  /// usually be seen in the "base" cases for more sophisticated approaches, for
  /// instance, when using warp shuffles on a GPU.
  Naive = 1,

  /// Each thread accumulates its contribution directly into the result variable
  /// using an atomic operation. This is the most naive approach possible, and
  /// is likely to have poor performance on both CPU and GPU. In certain cases,
  /// such as when the destination of a reduction in a loop is not loop
  /// invariant, this is the only strategy that will yield correct results. This
  /// is different from the \ref ThreadLocal strategy since it does not compute
  /// a partial result within a thread - all calls to the kit.reduce.0 intrinsic
  /// are essentially replaced with an atomic operation. This matters in cases
  /// where, for instance, a call to a reduce intrinsic is in a serial loop
  /// that is a child of a tapir loop. In such cases, reducing into a
  /// thread-local temporary may be beneficial, but, with this strategy, that
  /// will not be done.
  Direct = 2,

  /// Each thread computes its contribution to the reduction into a location in
  /// thread-local memory. This is then accumulated into the result variable
  /// using an atomic operation just before the thread finishes executing. This
  /// is the default strategy used for reductions by the CPU-centric tapir
  /// targets. It may also be used on the GPU, though it is not clear if there
  /// is any advantage to doing so. However, this can be more efficient than the
  /// \ref Direct strategy in cases where the call to the reduction intrinsic is
  /// in a serial loop within a tapir loop. If the destination of the reduction
  /// is loop-invariant relative to the parallel loop, accumulating the result
  /// into a thread-local variable first will usually require fewer atomic
  /// operations than the \ref Direct strategy.
  ThreadLocal = 3,

  /// Each thread in a block writes its contribution to the result into a
  /// dedicated location in GPU shared memory. These are then reduced into a
  /// single value for the block. This is accumulated into the result variable
  /// using an atomic operation.
  GPUSharedMemory = 4,

  /// Each thread in a warp computes its contribution to the result, which are
  /// the reduced to a single value for the warp using warp shuffles. A single
  /// thread in the warp will accumulate this directly into the result variable
  /// using an atomic operation.
  GPUWarpShuffle = 5,

  /// Each thread in a warp computes its contribution to the result, which are
  /// the reduced to a single value for the warp using warp shuffles. A single
  /// thread in the warp writes this to a dedicated location in shared memory.
  /// These are then reduced to a single value for the block. This value is
  /// accumulated into the result variable using an atomic operation.
  GPUWarpShuffleWithSharedMemory = 6,
};

/// Information about a reduction obtained by parsing a call to Kitsune's reduce
/// intrinsic. This is tightly coupled with the call itself. If the call object
/// is erased, this object will no longer be valid.
class ReductionInfo {
public:
  static constexpr unsigned argNumTTID = 0;
  static constexpr unsigned argNumReduceOp = 1;
  static constexpr unsigned argNumDest = 2;
  static constexpr unsigned argNumValue = 3;
  static constexpr unsigned argNumStrategy = 4;
  static constexpr unsigned argNumElemSize = 5;
  static constexpr unsigned argNumUnit = 6;
  static constexpr unsigned argNumReducer = 7;
  static constexpr unsigned argNumExtra = 8;

public:
  CallInst *call = nullptr;   ///< The call to a reduce intrinsic
  TTID tt;                    ///< The TTID of the tapir reduction loop
  ReduceOp reduceOp;          ///< The reduction operator
  ReductionStrategy strategy; ///< The reduction strategy to use
  unsigned elemSize = 0;      ///< The size (in bytes) of the reduced result

public:
  ReductionInfo(CallInst *call);

  /// Get the operand corresponding to the tapir target in the call.
  Value *getTTV() const { return call->getArgOperand(argNumTTID); }

  /// Get the operand corresponding to the reduce operator in the call.
  Value *getReduceOpV() const { return call->getArgOperand(argNumReduceOp); }

  /// Get the destination of the reduction.
  Value *getDest() const { return call->getArgOperand(argNumDest); }

  /// Get the value being reduced in the call
  Value *getValue() const { return call->getArgOperand(argNumValue); }

  /// Get the strategy used to perform this reduction.
  Value *getStrategy() const { return call->getArgOperand(argNumStrategy); }

  /// Get the operand corresponding to the element size in the call.
  Value *getElemSizeV() const { return call->getArgOperand(argNumElemSize); }

  /// Get the unit value for this reduction.
  Value *getUnit() const { return call->getArgOperand(argNumUnit); }

  /// Get the reducer for this reduction.
  Value *getReducer() const { return call->getArgOperand(argNumReducer); }

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
