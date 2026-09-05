//===- Reductions.cpp - Base types and utilities for reduction support ----===//
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

#include "kitsune/Core/Reductions.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/ModuleUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

using namespace llvm;

ReductionInfo::ReductionInfo(CallInst *call) : call(call) {
  tt = *fromConstant<TTID>(*cast<Constant>(call->getArgOperand(0)));
  reduceOp = *fromConstant<ReduceOp>(*cast<Constant>(call->getArgOperand(1)));
  elemSize = *fromConstant<unsigned>(*cast<Constant>(call->getArgOperand(3)));
}

SmallVector<Type *, 2> ReductionInfo::getOverloadTypes() const {
  Type *type = getValue()->getType();
  SmallVector<Type *, 2> overloadTypes = {type, type};
  for (Value *arg : getExtraArgs())
    overloadTypes.push_back(arg->getType());
  return overloadTypes;
}

SmallVector<Value *, 0> ReductionInfo::getExtraArgs() const {
  SmallVector<Value *, 0> extra;
  for (unsigned i = 7; i < call->arg_size(); ++i)
    extra.push_back(call->getArgOperand(i));
  return extra;
}

FunctionType *ReductionInfo::getReducerType() const {
  LLVMContext &ctx = call->getContext();
  Type *voidTy = Type::getVoidTy(ctx);
  SmallVector<Type *, 2> params = {getDest()->getType(), getValue()->getType()};
  for (Value *arg : getExtraArgs())
    params.push_back(arg->getType());
  return FunctionType::get(voidTy, params, /*isVarArg=*/false);
}

SmallVector<Value *, 2> ReductionInfo::getReducerArgs() const {
  SmallVector<Value *, 2> args = {getDest(), getValue()};
  for (Value *arg : getExtraArgs())
    args.push_back(arg);
  return args;
}

Type *ReductionInfo::getResultBufferType() const {
  Type *type = getType();
  if (!isa<PointerType>(type))
    return type;

  LLVMContext &ctx = type->getContext();
  Type *i8 = Type::getInt8Ty(ctx);
  return ArrayType::get(i8, elemSize);
};


// -----------------------------------------------------------------------------

template <> std::optional<ReduceOp> llvm::fromInt(int64_t i) {
  switch (i) {
  case 0: return ReduceOp::Custom;
  case 1: return ReduceOp::And;
  case 2: return ReduceOp::Or;
  case 3: return ReduceOp::Xor;
  case 5: return ReduceOp::Add;
  case 6: return ReduceOp::FAdd;
  case 7: return ReduceOp::Mul;
  case 8: return ReduceOp::FMul;
  case 16: return ReduceOp::FMax;
  case 17: return ReduceOp::FMaximum;
  case 18: return ReduceOp::FMaximumNum;
  case 20: return ReduceOp::FMin;
  case 21: return ReduceOp::FMinimum;
  case 22: return ReduceOp::FMinimumNum;
  case 24: return ReduceOp::SMax;
  case 25: return ReduceOp::SMin;
  case 26: return ReduceOp::UMax;
  case 27: return ReduceOp::UMin;
  default: return std::nullopt;
  }
}

template <> std::string llvm::toString(const ReduceOp &op) {
  switch (op) {
  case ReduceOp::Custom: return "custom";
  case ReduceOp::And: return "and";
  case ReduceOp::Or: return "or";
  case ReduceOp::Xor: return "xor";
  case ReduceOp::Add:
  case ReduceOp::FAdd: return "add";
  case ReduceOp::Mul:
  case ReduceOp::FMul: return "mul";
  case ReduceOp::FMax: return "fmax";
  case ReduceOp::SMax: return "smax";
  case ReduceOp::UMax: return "umax";
  case ReduceOp::FMaximum: return "maximum";
  case ReduceOp::FMaximumNum: return "maximumnum";
  case ReduceOp::FMin: return "fmin";
  case ReduceOp::SMin: return "smin";
  case ReduceOp::UMin: return "min";
  case ReduceOp::FMinimum: return "minimum";
  case ReduceOp::FMinimumNum: return "minimumnum";
  }
  llvm_unreachable("toString: Reduction operator not handled");
}

std::optional<AtomicRMWInst::BinOp> llvm::getAtomicOp(ReduceOp op) {
  // FIXME: AtomicRMWInst does not support FMaximumNum and FMinimumNum in
  // LLVM 21.x. When we upgrade to a newer version of LLVM that supports it,
  // this switch statement should be changed.
  switch (op) {
  case ReduceOp::And: return AtomicRMWInst::And;
  case ReduceOp::Or: return AtomicRMWInst::Or;
  case ReduceOp::Xor: return AtomicRMWInst::Xor;
  case ReduceOp::Add: return AtomicRMWInst::Add;
  case ReduceOp::FAdd: return AtomicRMWInst::FAdd;
  case ReduceOp::FMax: return AtomicRMWInst::FMax;
  case ReduceOp::FMaximum: return AtomicRMWInst::FMaximum;
  case ReduceOp::FMaximumNum: return std::nullopt;
  case ReduceOp::FMin: return AtomicRMWInst::FMin;
  case ReduceOp::FMinimum: return AtomicRMWInst::FMinimum;
  case ReduceOp::FMinimumNum: return std::nullopt;
  case ReduceOp::SMax: return AtomicRMWInst::Max;
  case ReduceOp::SMin: return AtomicRMWInst::Min;
  case ReduceOp::UMax: return AtomicRMWInst::UMax;
  case ReduceOp::UMin: return AtomicRMWInst::UMin;
  case ReduceOp::Custom:
  case ReduceOp::Mul:
  case ReduceOp::FMul: return std::nullopt;
  }
  llvm_unreachable("getAtomicOp: ReduceOp not handled");
}

static Constant *getOnes(Type *type) {
  return ConstantInt::get(type, -1, /*isSigned=*/false);
}

template <typename T> static T getMin() {
  return std::numeric_limits<T>::min();
}

static Constant *getMinInt(IntegerType *ity, bool isSigned) {
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

Constant *llvm::getUnitValue(ReduceOp op, Type *type) {
  switch (op) {
  case ReduceOp::Custom:
    llvm_unreachable("getUnitValueFor: cannot be used with ReduceOp::Custom");
  case ReduceOp::Or:
  case ReduceOp::Xor:
  case ReduceOp::Add:
  case ReduceOp::FAdd: return getZero(type);
  case ReduceOp::Mul:
  case ReduceOp::FMul: return getOne(type);
  case ReduceOp::And: return getOnes(type);
  case ReduceOp::FMin:
  case ReduceOp::FMinimum:
  case ReduceOp::FMinimumNum:
  case ReduceOp::SMin: return getMax(type, /*isSigned=*/true);
  case ReduceOp::UMin: return getMax(type, /*isSigned=*/false);
  case ReduceOp::FMax:
  case ReduceOp::FMaximum:
  case ReduceOp::FMaximumNum:
  case ReduceOp::SMax: return getMin(type, /*isSigned=*/true);
  case ReduceOp::UMax: return getMin(type, /*isSigned=*/false);
  }
  llvm_unreachable("getUnitValueFor: Reduction operand not handled");
}

static StringRef getReducerOpSuffix(ReduceOp op) {
  switch (op) {
  case ReduceOp::And: return ".and";
  case ReduceOp::Or: return ".or";
  case ReduceOp::Xor: return ".xor";
  case ReduceOp::Add:
  case ReduceOp::FAdd: return ".add";
  case ReduceOp::Mul:
  case ReduceOp::FMul: return ".mul";
  case ReduceOp::FMax:
  case ReduceOp::SMax:
  case ReduceOp::UMax: return ".max";
  case ReduceOp::FMaximum: return ".maximum";
  case ReduceOp::FMaximumNum: return ".maximumnum";
  case ReduceOp::FMin:
  case ReduceOp::SMin:
  case ReduceOp::UMin: return ".min";
  case ReduceOp::FMinimum: return ".minimum";
  case ReduceOp::FMinimumNum: return ".minimumnum";
  case ReduceOp::Custom:
    llvm_unreachable("getReducerOpSuffix: ReduceOp::Custom not expected");
  }
  llvm_unreachable("getReducerOpSuffix: ReduceOp not handled");
}

static StringRef getTypeSuffix(IntegerType *ity, bool isUnsigned) {
  switch (ity->getBitWidth()) {
  case 1: return ".i1";
  case 8: return isUnsigned ? ".u8" : ".i8";
  case 16: return isUnsigned ? ".u16" : ".i16";
  case 32: return isUnsigned ? ".u32" : ".i32";
  case 64: return isUnsigned ? ".u64" : ".i64";
  }
  llvm_unreachable("getReducerTypeSuffix: Bitwidth not handled");
}

static StringRef getTypeSuffix(Type *valueTy, bool isUnsigned) {
  if (auto *ity = dyn_cast<IntegerType>(valueTy))
    return getTypeSuffix(ity, isUnsigned);
  else if (valueTy->isFloatTy())
    return ".f32";
  else if (valueTy->isDoubleTy())
    return ".f64";
  llvm_unreachable("getTypeSuffix: Type not handled");
}

static StringRef getReducerTypeSuffix(ReduceOp op, Type *ty) {
  switch (op) {
  case ReduceOp::And:
  case ReduceOp::Or:
  case ReduceOp::Xor:
  case ReduceOp::UMax:
  case ReduceOp::UMin: return getTypeSuffix(ty, /*isUnsigned=*/true);
  case ReduceOp::Add:
  case ReduceOp::FAdd:
  case ReduceOp::Mul:
  case ReduceOp::FMul:
  case ReduceOp::SMax:
  case ReduceOp::FMax:
  case ReduceOp::FMaximum:
  case ReduceOp::FMaximumNum:
  case ReduceOp::SMin:
  case ReduceOp::FMin:
  case ReduceOp::FMinimum:
  case ReduceOp::FMinimumNum: return getTypeSuffix(ty, /*isUnsigned=*/false);
  case ReduceOp::Custom:
    llvm_unreachable("getReducerOpSuffix: ReduceOp::Custom not expected");
  }
  llvm_unreachable("getReducerTypeSuffix: ReduceOp not handled");
}

static std::string getReducerName(ReduceOp op, Type *ty) {
  std::string buf;
  raw_string_ostream os(buf);

  StringRef sfxOp = getReducerOpSuffix(op);
  StringRef sfxTy = getReducerTypeSuffix(op, ty);

  os << "__kitsune_reduce" << sfxOp << sfxTy;
  os.flush();

  return buf;
}

static Value *emitRHS(ReduceOp op, Value *curr, Value *v,
                      IRBuilder<> &builder) {
  auto createIntr = [&](Intrinsic::ID id, bool nsz = false) -> Instruction * {
    Type *ty = v->getType();
    auto *inst =
        cast<Instruction>(builder.CreateIntrinsic(id, {ty}, {curr, v}));
    if (nsz)
      inst->setHasNoSignedZeros(nsz);
    return inst;
  };

  switch (op) {
  case ReduceOp::And: return builder.CreateAnd(curr, v);
  case ReduceOp::Or: return builder.CreateOr(curr, v);
  case ReduceOp::Xor: return builder.CreateXor(curr, v);
  case ReduceOp::Add: return builder.CreateAdd(curr, v);
  case ReduceOp::FAdd: return builder.CreateFAdd(curr, v);
  case ReduceOp::Mul: return builder.CreateMul(curr, v);
  case ReduceOp::FMul: return builder.CreateFMul(curr, v);
  case ReduceOp::FMax: return createIntr(Intrinsic::maxnum, /*nsz=*/true);
  case ReduceOp::FMaximum: return createIntr(Intrinsic::maximum);
  case ReduceOp::FMaximumNum: return createIntr(Intrinsic::maximumnum);
  case ReduceOp::SMax: return createIntr(Intrinsic::smax);
  case ReduceOp::UMax: return createIntr(Intrinsic::umax);
  case ReduceOp::FMin: return createIntr(Intrinsic::minnum, /*nsz=*/true);
  case ReduceOp::FMinimum: return createIntr(Intrinsic::minimum);
  case ReduceOp::FMinimumNum: return createIntr(Intrinsic::minimumnum);
  case ReduceOp::SMin: return createIntr(Intrinsic::smin);
  case ReduceOp::UMin: return createIntr(Intrinsic::umin);
  case ReduceOp::Custom:
    llvm_unreachable("emitRHS: ReduceOp::Custom not expected");
  }
  llvm_unreachable("emitRHS: ReduceOp not handled");
}

Function *llvm::genReducer(ReduceOp op, Type *ty, Module &m) {
  std::string fname = getReducerName(op, ty);
  if (Function *f = m.getFunction(fname))
    return f;

  LLVMContext &ctx = m.getContext();
  Type *voidTy = Type::getVoidTy(ctx);
  Type *ptr = PointerType::getUnqual(ctx);

  Function *f = getOrInsertFunction(m, fname, voidTy, ptr, ty);
  Argument *res = f->getArg(0);
  Argument *v = f->getArg(1);

  BasicBlock *bb = BasicBlock::Create(ctx, "", f);
  IRBuilder<> builder(bb);

  Value *curr = builder.CreateLoad(ty, res);
  Value *nw = emitRHS(op, curr, v, builder);
  builder.CreateStore(nw, res);
  builder.CreateRetVoid();

  res->setName("res");
  v->setName("v");

  // This function may be regenerated in multiple translation units and linked
  // together. linkonce_odr allows inlining, whereas linkonce does not.
  f->setLinkage(GlobalValue::LinkOnceODRLinkage);

  return f;
}

SmallVector<ReductionInfo, 1> llvm::collectReductions(Loop &loop) {
  SmallVector<ReductionInfo, 1> reductions;
  for (BasicBlock *bb : loop.getBlocks())
    for (Instruction &inst : *bb)
      if (auto *call = dyn_cast<CallInst>(&inst))
        if (call->getIntrinsicID() == Intrinsic::kit_reduce_0)
          reductions.emplace_back(call);
  return reductions;
}
