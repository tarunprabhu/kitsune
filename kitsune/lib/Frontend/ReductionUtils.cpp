//===- Reductionutils.cpp - Utilities for reduction builtins --------------===//
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

#include "kitsune/Frontend/ReductionUtils.h"
#include "llvm/IR/Constants.h"

using namespace llvm;

template <> std::optional<ReduceOp> llvm::fromInt(int64_t i) {
  // clang-format off
  switch (i) {
  case 0: return ReduceOp::Custom;
  case 1: return ReduceOp::BAnd;
  case 2: return ReduceOp::BOr;
  case 3: return ReduceOp::BXor;
  case 4: return ReduceOp::LAnd;
  case 5: return ReduceOp::LOr;
  case 6: return ReduceOp::LXor;
  case 7: return ReduceOp::Max;
  case 8: return ReduceOp::MaxLoc;
  case 9: return ReduceOp::Min;
  case 10: return ReduceOp::MinLoc;
  case 11: return ReduceOp::Prod;
  case 12: return ReduceOp::Sum;
  default: return std::nullopt;
  }
  // clang-format on
}

StringRef llvm::toString(ReduceOp op) {
  // clang-format off
  switch (op) {
  case ReduceOp::Custom: return "custom";
  case ReduceOp::BAnd: return "bitwise and";
  case ReduceOp::BOr: return "bitwise or";
  case ReduceOp::BXor: return "bitwise xor";
  case ReduceOp::LAnd: return "logical and";
  case ReduceOp::LOr: return "logical or";
  case ReduceOp::LXor: return "logical xor";
  case ReduceOp::Max: return "max";
  case ReduceOp::MaxLoc: return "maxloc";
  case ReduceOp::Min: return "min";
  case ReduceOp::MinLoc: return "minloc";
  case ReduceOp::Prod: return "product";
  case ReduceOp::Sum: return "sum";
  }
  // clang-format on
  llvm_unreachable("toString(ReduceOp): Reduction operator not handled");
}

static Constant *getZero(Type *type) {
  if (type->isIntegerTy())
    return ConstantInt::get(type, 0, /*isSigned=*/false);
  else if (type->isFloatTy() || type->isDoubleTy())
    return ConstantFP::get(type, 0);
  llvm_unreachable("getZero: Type not handled");
}

static Constant *getOne(Type *type) {
  if (type->isIntegerTy())
    return ConstantInt::get(type, 1, /*isSigned=*/false);
  else if (type->isFloatTy())
    return ConstantFP::get(type, 1.0f);
  else if (type->isDoubleTy())
    return ConstantFP::get(type, 1.0);
  llvm_unreachable("getZero: Type not handled");
}

static Constant *getOnes(Type *type) {
  return ConstantInt::get(type, -1, /*isSigned=*/false);
}

template <typename T> static T getMin() {
  return std::numeric_limits<T>::min();
}

static Constant *getMinInt(IntegerType *ity, bool isSigned) {
  // clang-format off
  if (isSigned) {
    switch (ity->getBitWidth()) {
    case 8: return ConstantInt::get(ity, getMin<int8_t>(), isSigned);
    case 16: return ConstantInt::get(ity, getMin<int16_t>(), isSigned);
    case 32: return ConstantInt::get(ity, getMin<int32_t>(), isSigned);
    case 64: return ConstantInt::get(ity, getMin<int64_t>(), isSigned);
    default: break;
    }
  } else {
    switch (ity->getBitWidth()) {
    case 8: return ConstantInt::get(ity, uint8_t(0), isSigned);
    case 16: return ConstantInt::get(ity, uint16_t(0), isSigned);
    case 32: return ConstantInt::get(ity, uint32_t(0), isSigned);
    case 64: return ConstantInt::get(ity, uint64_t(0), isSigned);
    default: break;
    }
  }
  // clang-format on
  llvm_unreachable("getMinInt: Bitwidth not handled");
}

static Constant *getMin(Type *type, bool isSigned) {
  LLVMContext &ctx = type->getContext();
  if (auto *ity = dyn_cast<IntegerType>(type))
    return getMinInt(ity, isSigned);
  else if (type->isFloatTy())
    return ConstantFP::get(ctx, APFloat(std::numeric_limits<float>::min()));
  else if (type->isDoubleTy())
    return ConstantFP::get(ctx, APFloat(std::numeric_limits<double>::min()));
  llvm_unreachable("getMin: Type not handled");
}

template <typename T> static T getMax() {
  return std::numeric_limits<T>::max();
}

static Constant *getMaxInt(IntegerType *ity, bool isSigned) {
  // clang-format off
  if (isSigned) {
    switch (ity->getBitWidth()) {
    case 8: return ConstantInt::get(ity, getMax<int8_t>(), isSigned);
    case 16: return ConstantInt::get(ity, getMax<int16_t>(), isSigned);
    case 32: return ConstantInt::get(ity, getMax<int32_t>(), isSigned);
    case 64: return ConstantInt::get(ity, getMax<int64_t>(), isSigned);
    default: break;
    }
  } else {
    switch (ity->getBitWidth()) {
    case 8: return ConstantInt::get(ity, getMax<uint8_t>(), isSigned);
    case 16: return ConstantInt::get(ity, getMax<uint16_t>(), isSigned);
    case 32: return ConstantInt::get(ity, getMax<uint32_t>(), isSigned);
    case 64: return ConstantInt::get(ity, getMax<uint64_t>(), isSigned);
    default: break;
    }
  }
  // clang-format on
  llvm_unreachable("getMaxInt: Bitwidth not handled");
}

static Constant *getMax(Type *type, bool isSigned) {
  LLVMContext &ctx = type->getContext();
  if (auto *ity = dyn_cast<IntegerType>(type))
    return getMaxInt(ity, isSigned);
  else if (type->isFloatTy())
    return ConstantFP::get(ctx, APFloat(std::numeric_limits<float>::max()));
  else if (type->isDoubleTy())
    return ConstantFP::get(ctx, APFloat(std::numeric_limits<double>::max()));
  llvm_unreachable("getMax: Type not handled");
}

Constant *llvm::getUnitValueFor(ReduceOp op, Type *type, bool isSigned) {
  switch (op) {
  case ReduceOp::Custom:
    llvm_unreachable("getUnitValueFor: cannot be used with ReduceOp::Custom");
  case ReduceOp::BOr:
  case ReduceOp::BXor:
  case ReduceOp::LOr:
  case ReduceOp::LXor:
  case ReduceOp::Sum:
    return getZero(type);
  case ReduceOp::LAnd:
  case ReduceOp::Prod:
    return getOne(type);
  case ReduceOp::BAnd:
    return getOnes(type);
  case ReduceOp::Min:
  case ReduceOp::MinLoc:
    return getMax(type, isSigned);
  case ReduceOp::Max:
  case ReduceOp::MaxLoc:
    return getMin(type, isSigned);
  }
  llvm_unreachable("getUnitValueFor: Reduction operand not handled");
}

Constant *llvm::getUnitValueFor(ReduceOp op, Type *type) {
  return getUnitValueFor(op, type, /*isSigned=*/false);
}
