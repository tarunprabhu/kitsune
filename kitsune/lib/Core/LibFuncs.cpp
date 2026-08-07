//===- LibFuncs.cpp - Utilities for Kitsune's library functions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to work with Kitsune's library functions. These are mainly useful
// when lowering Kitsune's intrinsics, but they can be used whenever these
// library functions need to be used directly in IR.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/LibFuncs.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {

// The types that may appear in the signatures of Kitsune's library functions.
enum class CType {
#define GET_CTYPE_ENUMS
#include "kitsune/Core/LibFuncs.inc"
};

} // namespace

static Type *getLLVMType(CType ctype, LLVMContext &ctx) {
  switch (ctype) {
  case CType::Void:
    return Type::getVoidTy(ctx);
  case CType::Bool:
    return Type::getInt8Ty(ctx);
  case CType::Ptr:
    return PointerType::getUnqual(ctx);
  case CType::I8:
  case CType::U8:
    return Type::getInt8Ty(ctx);
  case CType::I16:
  case CType::U16:
    return Type::getInt16Ty(ctx);
  case CType::I32:
  case CType::U32:
    return Type::getInt32Ty(ctx);
  case CType::I64:
  case CType::U64:
    return Type::getInt64Ty(ctx);
  case CType::Float:
    return Type::getFloatTy(ctx);
  case CType::Double:
    return Type::getDoubleTy(ctx);
  case CType::VarArgs:
    return nullptr;
  }
  llvm_unreachable("getLLVMType: CType not handled");
}

// Get the function type from the given types. The first element of \p ctypes is
// the return type. The remaining parameter types. Vararg functions are not
// supported.
static FunctionType *getLibFuncType(ArrayRef<CType> ctypes, LLVMContext &ctx) {
  bool isVarArg = false;
  Type *ret = getLLVMType(ctypes[0], ctx);
  SmallVector<Type *, 4> params;
  for (CType param : ctypes.drop_front()) {
    // getLLVMType will return null if the type is CType::VarArgs. Tablgen will
    // have ensured that a VarArg type is the last type in the specification.
    if (Type *paramTy = getLLVMType(param, ctx))
      params.push_back(paramTy);
    else
      isVarArg = true;
  }
  return FunctionType::get(ret, params, isVarArg);
}

// Because of the way the LIBFUNC macros are emitted, we are guaranteed to have
// at least one type for every library function - the return type. We,
// therefore, don't need to handle the case where 0 types are provided to
// CTYPES. Which is just as well since I couldn't get this ugly sequence working
// for that case anyway.
//
// Hopefully, support for 32 parameters is sufficient.
#define CTYPE(T) CType::T
#define C1(T) CTYPE(T)
#define C2(T, ...) CTYPE(T), C1(__VA_ARGS__)
#define C3(T, ...) CTYPE(T), C2(__VA_ARGS__)
#define C4(T, ...) CTYPE(T), C3(__VA_ARGS__)
#define C5(T, ...) CTYPE(T), C4(__VA_ARGS__)
#define C6(T, ...) CTYPE(T), C5(__VA_ARGS__)
#define C7(T, ...) CTYPE(T), C6(__VA_ARGS__)
#define C8(T, ...) CTYPE(T), C7(__VA_ARGS__)
#define C9(T, ...) CTYPE(T), C8(__VA_ARGS__)
#define C10(T, ...) CTYPE(T), C9(__VA_ARGS__)
#define C11(T, ...) CTYPE(T), C10(__VA_ARGS__)
#define C12(T, ...) CTYPE(T), C11(__VA_ARGS__)
#define C13(T, ...) CTYPE(T), C12(__VA_ARGS__)
#define C14(T, ...) CTYPE(T), C13(__VA_ARGS__)
#define C15(T, ...) CTYPE(T), C14(__VA_ARGS__)
#define C16(T, ...) CTYPE(T), C15(__VA_ARGS__)
#define C17(T, ...) CTYPE(T), C16(__VA_ARGS__)
#define C18(T, ...) CTYPE(T), C17(__VA_ARGS__)
#define C19(T, ...) CTYPE(T), C18(__VA_ARGS__)
#define C20(T, ...) CTYPE(T), C19(__VA_ARGS__)
#define C21(T, ...) CTYPE(T), C20(__VA_ARGS__)
#define C22(T, ...) CTYPE(T), C21(__VA_ARGS__)
#define C23(T, ...) CTYPE(T), C22(__VA_ARGS__)
#define C24(T, ...) CTYPE(T), C23(__VA_ARGS__)
#define C25(T, ...) CTYPE(T), C24(__VA_ARGS__)
#define C26(T, ...) CTYPE(T), C25(__VA_ARGS__)
#define C27(T, ...) CTYPE(T), C26(__VA_ARGS__)
#define C28(T, ...) CTYPE(T), C27(__VA_ARGS__)
#define C29(T, ...) CTYPE(T), C28(__VA_ARGS__)
#define C30(T, ...) CTYPE(T), C29(__VA_ARGS__)
#define C31(T, ...) CTYPE(T), C30(__VA_ARGS__)
#define C32(T, ...) CTYPE(T), C31(__VA_ARGS__)
#define C33(T, ...) CTYPE(T), C32(__VA_ARGS__)

#define CTYPES_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14,   \
                _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26,    \
                _27, _28, _29, _30, _31, _32, _33, F, ...)                     \
  F

// Prior to C++20, passing no arguments to ... was not allowed. The X is added
// at the end of the CTYPES_ call to ensure that an argument is always passed to
// the ellipses in CTYPES_.
#define CTYPES(...)                                                            \
  CTYPES_(__VA_ARGS__, C33, C32, C31, C30, C29, C28, C27, C26, C25, C24, C23,  \
          C22, C21, C20, C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, \
          C8, C7, C6, C5, C4, C3, C2, C1, X)(__VA_ARGS__)

StringRef llvm::getLibFuncName(KitFunc f) {
  switch (f) {
#define LIBFUNC(NAME, LINKAGE_NAME, ...)                                       \
  case KitFunc::NAME:                                                          \
    return LINKAGE_NAME;
#define GET_LIBFUNCS
#include "kitsune/Core/LibFuncs.inc"
  }
  llvm_unreachable("getLibFuncName: KitFunc not handled");
}

FunctionType *llvm::getLibFuncType(KitFunc f, LLVMContext &ctx) {
  switch (f) {
#define GET_LIBFUNCS
#define LIBFUNC(NAME, LINKAGE_NAME, ...)                                       \
  case KitFunc::NAME: {                                                        \
    CType types[] = {CTYPES(__VA_ARGS__)};                                     \
    return ::getLibFuncType(types, ctx);                                       \
  }
#include "kitsune/Core/LibFuncs.inc"
  }
  llvm_unreachable("getLibFuncType: KitFunc not handled");
}

Function *llvm::getDeclarationIfExists(Module &m, KitFunc f) {
  return m.getFunction(getLibFuncName(f));
}

FunctionCallee llvm::getOrInsertLibFunc(Module &m, KitFunc libFunc) {
  StringRef funcName = getLibFuncName(libFunc);
  if (Function *f = m.getFunction(funcName))
    return FunctionCallee(f);

  LLVMContext &ctx = m.getContext();
  FunctionType *funcTy = getLibFuncType(libFunc, ctx);
  FunctionCallee callee = m.getOrInsertFunction(funcName, funcTy);
  Function *f = cast<Function>(callee.getCallee());

  // Even if we know that the function eventually calls free, we still set this
  // attribute because that is what LLVM does as well.
  // TODO: Figure out if there is something smarter that can be done here.
  f->setDoesNotFreeMemory();
  f->setDoesNotThrow();
  f->setMemoryEffects(MemoryEffects::inaccessibleOrArgMemOnly());
  f->setWillReturn();

  return callee;
}
