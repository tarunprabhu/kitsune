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
#include "kitsune/Clang/ReductionUtils.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/KitOptions.h"
#include "kitsune/Support/AddrSpace.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

// TODO:? Is it worth making this available as an expression utility?
// Strip implicit expressions and cleanups to retrieve the underlying
// expression.
static const Expr *getUnderlyingExpr(const Expr *expr) {
  const Expr *underlying = expr->IgnoreImplicit();
  if (const auto *ewc = dyn_cast<ExprWithCleanups>(underlying))
    return getUnderlyingExpr(ewc->getSubExpr());
  return underlying;
}

// TODO:? Is it worth making this available as an expression utility?
// Get the underlying unqualified desugared type of the expression
static const clang::Type *getUnqualifiedDesugaredType(const Expr &expr) {
  return expr.getType()->getUnqualifiedDesugaredType();
}

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

static StringRef getReducerTypeSuffix(llvm::IntegerType *ity, bool isUnsigned) {
  // clang-format off
  switch (ity->getBitWidth()) {
  case 1: return ".i1";
  case 8: return isUnsigned ? ".u8" : ".i8";
  case 16: return isUnsigned ? ".u16" : ".i16";
  case 32: return isUnsigned ? ".u32" : ".i32";
  case 64: return isUnsigned ? ".u64" : ".i64";
  }
  // clang-format on
  llvm_unreachable("getReducerTypeSuffix: Bitwidth not handled");
}

static StringRef getReducerTypeSuffix(llvm::Type *valueTy, bool isUnsigned) {
  if (auto *ity = dyn_cast<IntegerType>(valueTy))
    return getReducerTypeSuffix(ity, isUnsigned);
  else if (valueTy->isFloatTy())
    return ".f32";
  else if (valueTy->isDoubleTy())
    return ".f64";
  llvm_unreachable("getReducerTypeSuffix: Type not handled");
}

static StringRef getReducerOpSuffix(ReduceOp op) {
  // clang-format off
  switch (op) {
  case ReduceOp::Custom:
    llvm_unreachable("getReducerOpSuffix not expected with custom reduce");
  case ReduceOp::BAnd: return ".band";
  case ReduceOp::BOr: return ".bor";
  case ReduceOp::BXor: return ".bxor";
  case ReduceOp::LAnd: return ".land";
  case ReduceOp::LOr: return ".lor";
  case ReduceOp::LXor: return ".lxor";
  case ReduceOp::Max: return ".max";
  case ReduceOp::MaxLoc: return ".maxloc";
  case ReduceOp::Min: return ".min";
  case ReduceOp::MinLoc: return ".minloc";
  case ReduceOp::Sum: return ".sum";
  case ReduceOp::Prod: return ".prod";
  }
  // clang-format on
  llvm_unreachable("getReducerOpSuffix: Reducer op not handled");
}

static std::string getReducerName(ReduceOp op, llvm::Type *valueTy,
                                  bool isUnsigned) {
  // For the min and max reductions, the sign matters because it affects the
  // unit value for the reduction. For the other types, the sign of the type
  // is not relevant when generating LLVM IR.
  auto isSignSignificant = [](ReduceOp op) -> bool {
    return op == ReduceOp::Min || op == ReduceOp::Max;
  };

  std::string buf;
  raw_string_ostream os(buf);
  StringRef sfxOp = getReducerOpSuffix(op);
  StringRef sfxTy = isSignSignificant(op)
                        ? getReducerTypeSuffix(valueTy, isUnsigned)
                        : getReducerTypeSuffix(valueTy, /*isUnsigned=*/false);

  os << "__kitsune_reduce" << sfxOp << sfxTy;
  os.flush();

  return buf;
}

static Value *emitSum(Value *op1, Value *op2, IRBuilder<> &builder) {
  llvm::Type *type = op1->getType();
  if (type->isFloatingPointTy())
    return builder.CreateFAdd(op1, op2);
  else if (type->isIntegerTy())
    return builder.CreateAdd(op1, op2);
  llvm_unreachable("emitAdd: Type not handled");
}

static Value *emitProd(Value *op1, Value *op2, IRBuilder<> &builder) {
  llvm::Type *type = op1->getType();
  if (type->isFloatingPointTy())
    return builder.CreateFMul(op1, op2);
  else if (type->isIntegerTy())
    return builder.CreateMul(op1, op2);
  llvm_unreachable("emitProd: Type not handled");
}

static Value *emitMin(Value *op1, Value *op2, bool isUnsigned,
                      IRBuilder<> &builder) {
  auto getIntrinsic = [](llvm::Type *type, bool isUnsigned) -> Intrinsic::ID {
    if (type->isIntegerTy() && isUnsigned)
      return Intrinsic::umin;
    else if (type->isIntegerTy())
      return Intrinsic::smin;
    else if (type->isFloatingPointTy())
      return Intrinsic::minimum;
    llvm_unreachable("emitMin: Type not handled");
  };

  llvm::Type *type = op1->getType();
  Intrinsic::ID id = getIntrinsic(type, isUnsigned);
  return builder.CreateIntrinsic(id, {type}, {op1, op2});
}

static Value *emitMax(Value *op1, Value *op2, bool isUnsigned,
                      IRBuilder<> &builder) {
  auto getIntrinsic = [](llvm::Type *type, bool isUnsigned) -> Intrinsic::ID {
    if (type->isIntegerTy() && isUnsigned)
      return Intrinsic::umax;
    else if (type->isIntegerTy())
      return Intrinsic::smax;
    else if (type->isFloatingPointTy())
      return Intrinsic::maximum;
    llvm_unreachable("emitMin: Type not handled");
  };

  llvm::Type *type = op1->getType();
  Intrinsic::ID id = getIntrinsic(type, isUnsigned);
  return builder.CreateIntrinsic(id, {type}, {op1, op2});
}

static Value *emitRhs(ReduceOp op, Value *op1, Value *op2, bool isUnsigned,
                      IRBuilder<> &builder) {
  // clang-format off
  switch (op) {
  case ReduceOp::Custom:
  case ReduceOp::MaxLoc:
  case ReduceOp::MinLoc:
    llvm_unreachable("NOT YET IMPLEMENTED: emitRhs for reduction operator");
  case ReduceOp::LAnd: return builder.CreateAnd(op1, op2);
  case ReduceOp::LOr: return builder.CreateOr(op1, op2);
  case ReduceOp::LXor: return builder.CreateXor(op1, op2);
  case ReduceOp::BAnd: return builder.CreateAnd(op1, op2);
  case ReduceOp::BOr: return builder.CreateOr(op1, op2);
  case ReduceOp::BXor: return builder.CreateXor(op1, op2);
  case ReduceOp::Max: return emitMax(op1, op2, isUnsigned, builder);
  case ReduceOp::Min: return emitMin(op1, op2, isUnsigned, builder);
  case ReduceOp::Sum: return emitSum(op1, op2, builder);
  case ReduceOp::Prod: return emitProd(op1, op2, builder);
  }
  // clang-format on
  llvm_unreachable("emitRhs: Reducer op not handled");
}

static Function *emitKitReducerDefn(llvm::Module &m, ReduceOp op,
                                    llvm::Type *valueTy, bool isUnsigned) {
  LLVMContext &ctx = m.getContext();
  llvm::Type *voidTy = llvm::Type::getVoidTy(ctx);
  llvm::Type *ptr = llvm::PointerType::getUnqual(ctx);
  llvm::FunctionType *fty =
      llvm::FunctionType::get(voidTy, {ptr, valueTy}, /*isVarArg=*/false);
  std::string fname = getReducerName(op, valueTy, isUnsigned);

  FunctionCallee callee = m.getOrInsertFunction(fname, fty);
  Function *f = cast<Function>(callee.getCallee());
  if (f->size())
    return f;

  BasicBlock *bb = BasicBlock::Create(ctx, "", f);
  IRBuilder<> builder(bb);
  Value *curr = builder.CreateLoad(valueTy, f->getArg(0));
  Value *upd = emitRhs(op, curr, f->getArg(1), isUnsigned, builder);
  (void)builder.CreateStore(upd, f->getArg(0));
  (void)builder.CreateRetVoid();

  // This function may be regenerated in multiple translation units and linked
  // together. linkonce_odr allows inlining, whereas linkonce does not.
  f->setLinkage(GlobalValue::LinkOnceODRLinkage);

  return f;
}

static RValue emitKitReduceCall(const CallExpr &theCall, CodeGenFunction &cgf,
                                ReturnValueSlot rv) {
  auto getReduceOp = [](Value *v) -> ReduceOp {
    return *fromInt<ReduceOp>(cast<ConstantInt>(v)->getLimitedValue());
  };

  CodeGenModule &cgm = cgf.CGM;
  const driver::KitOptions &kitOpts = cgm.getKitOpts();

  // FIXME: This should not be mandatory. But for now, we leave it this way.
  // There is a broader question of how to handle Kitsune's builtins when the
  // kitcc frontend is used but a tapir target is not provided. Until that is
  // resolved, we leave this in and accept that this might fail in some cases.
  assert(kitOpts.hasTTID() && "TTID not set in Kitsune options");

  const Expr *destExpr = theCall.getArg(0);
  const Expr *opExpr = theCall.getArg(1);
  const Expr *valueExpr = getUnderlyingExpr(theCall.getArg(2));

  LLVMContext &ctx = cgm.getLLVMContext();
  const DataLayout &layout = cgm.getDataLayout();
  llvm::Module &m = cgm.getModule();
  bool isUnsigned =
      getUnqualifiedDesugaredType(*valueExpr)->isUnsignedIntegerType();

  Value *tt = toConstant(*kitOpts.getTTID(), ctx);
  Value *dest = cgf.EmitScalarExpr(destExpr);
  Value *op = cgf.EmitScalarExpr(opExpr);
  Value *value = cgf.EmitScalarExpr(valueExpr);
  llvm::Type *valueTy = value->getType();
  unsigned valueSz = layout.getTypeStoreSize(valueTy);
  Value *valueSize = toConstant(valueSz, ctx);
  ReduceOp reduceOp = getReduceOp(op);
  Value *unit = getUnitValueFor(reduceOp, valueTy, !isUnsigned);
  Function *reducer = emitKitReducerDefn(m, reduceOp, valueTy, isUnsigned);

  CGBuilderTy &builder = cgf.Builder;
  Value *call =
      builder.CreateIntrinsic(Intrinsic::kit_reduce_0, {valueTy, valueTy},
                              {tt, dest, valueSize, value, unit, reducer});
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
