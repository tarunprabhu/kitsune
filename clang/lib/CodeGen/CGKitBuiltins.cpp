//===- CGKitBuiltins.cpp - Codegen for Kitsune's builtins -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// "Codegen" (i.e. LLVM IR generation) for Kitsune's builtins
//
//===----------------------------------------------------------------------===//

#include "CGKitsune.h"
#include "CodeGenFunction.h"
#include "kitsune/Clang/ASTUtils.h"
#include "kitsune/Core/AddrSpace.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/KitOptions.h"
#include "kitsune/Core/Reductions.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

static RValue emitKitMobileAllocCall(const CallExpr &theCall,
                                     CodeGenFunction &cgf, ReturnValueSlot rv) {
  CodeGenModule &cgm = cgf.CGM;
  const driver::KitOptions &kitOpts = cgm.getKitOpts();

  CGBuilderTy &builder = cgf.Builder;
  LLVMContext &ctx = builder.getContext();

  Function *f = cgm.getIntrinsic(Intrinsic::kit_mobile_alloc);
  llvm::FunctionType *fty = f->getFunctionType();
  Value *size = cgf.EmitScalarExpr(theCall.getArg(0));
  if (size->getType() != fty->getParamType(1))
    size = builder.CreateTruncOrBitCast(size, fty->getParamType(1));

  // FIXME: Defaulting to serial has the effect of behaving similar to vanilla
  // C/C++. It is reasonable for this, but we need to consider what to do when
  // --tapir is not provided with Kitsune's frontend.
  Value *tt = toConstant(kitOpts.getTTIDOr(TTID::Serial), ctx);

  return RValue::get(builder.CreateCall(f, {tt, size}));
}

static RValue emitKitMobileFreeCall(const CallExpr &theCall,
                                    CodeGenFunction &cgf, ReturnValueSlot rv) {
  CodeGenModule &cgm = cgf.CGM;
  const driver::KitOptions &kitOpts = cgm.getKitOpts();

  CGBuilderTy &builder = cgf.Builder;
  LLVMContext &ctx = builder.getContext();

  // FIXME: Defaulting to serial has the effect of behaving similar to vanilla
  // C/C++. It is reasonable for this, but we need to consider what to do when
  // --tapir is not provided with Kitsune's frontend.
  Value *tt = toConstant(kitOpts.getTTIDOr(TTID::Serial), ctx);
  Value *ptr = cgf.EmitScalarExpr(theCall.getArg(0));

  return RValue::get(
      builder.CreateIntrinsic(Intrinsic::kit_mobile_free, {tt, ptr}));
}

static RValue emitKitMobileCastUnsafeCall(const CallExpr &theCall,
                                          CodeGenFunction &cgf,
                                          ReturnValueSlot rv) {
  CGBuilderTy &builder = cgf.Builder;
  LLVMContext &ctx = builder.getContext();

  Value *ptr = cgf.EmitScalarExpr(theCall.getArg(0));
  llvm::Type *destTy = llvm::PointerType::get(ctx, KitAS::Mobile);

  return RValue::get(builder.CreateAddrSpaceCast(ptr, destTy));
}

static ReduceOp getReduceOp(Value *v, bool isFloat, bool isUnsigned) {
  switch (cast<ConstantInt>(v)->getLimitedValue()) {
  case 0: // KIT_CUSTOM
    return ReduceOp::Custom;
  case 1: // KIT_AND
    return ReduceOp::And;
  case 2: // KIT_OR
    return ReduceOp::Or;
  case 3: // KIT_XOR
    return ReduceOp::Xor;
  case 5: // KIT_ADD
    return isFloat ? ReduceOp::FAdd : ReduceOp::Add;
  case 7: // KIT_MUL
    return isFloat ? ReduceOp::FMul : ReduceOp::Mul;
  case 16: // KIT_MAX
    if (isFloat)
      return ReduceOp::FMax;
    else if (isUnsigned)
      return ReduceOp::UMax;
    else
      return ReduceOp::SMax;
  case 17: // KIT_MAXIMUM
    return ReduceOp::FMaximum;
  case 18: // KIT_MAXIMUM_NUM
    return ReduceOp::FMaximumNum;
  case 20: // KIT_MIN
    if (isFloat)
      return ReduceOp::FMin;
    else if (isUnsigned)
      return ReduceOp::UMin;
    else
      return ReduceOp::SMin;
  case 21: // KIT_MINIMUM
    return ReduceOp::FMinimum;
  case 22: // KIT_MINIMUM_NUM
    return ReduceOp::FMinimumNum;
  default:
    llvm_unreachable("getReduceOp: Unknown builtin operator");
  }
}

Value *getReducer(ReduceOp op, llvm::Type *ty, llvm::Module &m) {
  if (op == ReduceOp::Custom)
    llvm_unreachable("getReducer: ReduceOp::Custom not yet implemented");
  return genReducer(op, ty, m);
}

static RValue emitKitReduceCall(const CallExpr &theCall, CodeGenFunction &cgf,
                                ReturnValueSlot rv) {
  CodeGenModule &cgm = cgf.CGM;
  const driver::KitOptions &kitOpts = cgm.getKitOpts();

  // FIXME: This should not be mandatory. But for now, we leave it this way.
  // There is a broader question of how to handle Kitsune's builtins when the
  // kitcc frontend is used but a tapir target is not provided. Until that is
  // resolved, we leave this in and accept that this might fail in some cases.
  assert(kitOpts.hasTTID() && "TTID not set in Kitsune options");

  LLVMContext &ctx = cgm.getLLVMContext();
  const DataLayout &layout = cgm.getDataLayout();
  llvm::Module &m = cgm.getModule();

  const Expr *destExpr = theCall.getArg(0);
  const Expr *opExpr = theCall.getArg(1);
  const Expr *valueExpr = getUnderlyingExpr(theCall.getArg(2));

  Value *tt = toConstant(*kitOpts.getTTID(), ctx);
  Value *dest = cgf.EmitScalarExpr(destExpr);
  Value *op = cgf.EmitScalarExpr(opExpr);
  Value *value = cgf.EmitScalarExpr(valueExpr);

  llvm::Type *valueTy = value->getType();
  bool isFloat = valueTy->isFloatingPointTy();
  bool isUnsigned =
      getUnqualifiedDesugaredType(valueExpr)->isUnsignedIntegerType();

  ReduceOp reduceOp = getReduceOp(op, isFloat, isUnsigned);
  Value *valueSz = toConstant((uint32_t)layout.getTypeStoreSize(valueTy), ctx);
  Value *unit = getUnitValue(reduceOp, valueTy);
  Value *reducer = getReducer(reduceOp, valueTy, m);
  Value *reduceOpV = toConstant((uint32_t)reduceOp, ctx);

  CGBuilderTy &builder = cgf.Builder;
  Value *call = builder.CreateIntrinsic(
      Intrinsic::kit_reduce_0, {valueTy, valueTy},
      {tt, reduceOpV, dest, valueSz, value, unit, reducer});

  return RValue::get(call);
}

RValue clang::CodeGen::EmitKitBuiltinCall(CodeGenFunction &cgf,
                                          const FunctionDecl *funcDecl,
                                          unsigned builtinID,
                                          const CallExpr *theCall,
                                          ReturnValueSlot rv) {
  switch (builtinID) {
  case Builtin::BIkitsune_mobile_alloc:
    return emitKitMobileAllocCall(*theCall, cgf, rv);
  case Builtin::BIkitsune_mobile_free:
    return emitKitMobileFreeCall(*theCall, cgf, rv);
  case Builtin::BI__kitsune_mobile_cast_unsafe:
    return emitKitMobileCastUnsafeCall(*theCall, cgf, rv);
  case Builtin::BI__kitsune_reduce:
    return emitKitReduceCall(*theCall, cgf, rv);
  default:
    llvm_unreachable("EmitKitBuiltinExpr: BuiltinID not handled");
  }
}

bool clang::CodeGen::IsKitBuiltin(unsigned builtinID) {
  switch (builtinID) {
  case Builtin::BIkitsune_mobile_alloc:
  case Builtin::BIkitsune_mobile_free:
  case Builtin::BI__kitsune_mobile_cast_unsafe:
  case Builtin::BI__kitsune_reduce:
    return true;
  default:
    return false;
  }
}
