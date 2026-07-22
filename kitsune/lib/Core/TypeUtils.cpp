//===- TypeUtils.cpp - Helper functions for LLVM's types ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of utilities for LLVM's types
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TypeUtils.h"
#include "kitsune/Core/AddrSpace.h"
#include "kitsune/Support/TypeTraits.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

using namespace llvm;

bool llvm::isByteArrayTy(Type *Ty) {
  if (auto *arrayTy = dyn_cast<ArrayType>(Ty))
    return arrayTy->getElementType()->isIntegerTy(8);
  return false;
}

bool llvm::isMobilePointerTy(Type *ty) {
  if (auto *ptrTy = dyn_cast<PointerType>(ty))
    if (ptrTy->getAddressSpace() == KitAS::Mobile)
      return true;
  return false;
}

template <typename T, std::enable_if_t<std::is_bool_v<T>, int> = 0>
static Type *getLLVMTypeImpl(LLVMContext &ctx) {
  return Type::getInt1Ty(ctx);
}

template <typename T, std::enable_if_t<std::is_integer_v<T>, int> = 0>
static Type *getLLVMTypeImpl(LLVMContext &ctx) {
  return IntegerType::get(ctx, sizeof(T) * 8);
}

template <typename T, std::enable_if_t<std::is_float_v<T>, int> = 0>
static Type *getLLVMTypeImpl(LLVMContext &ctx) {
  return Type::getFloatTy(ctx);
}

template <typename T, std::enable_if_t<std::is_double_v<T>, int> = 0>
static Type *getLLVMTypeImpl(LLVMContext &ctx) {
  return Type::getDoubleTy(ctx);
}

template <typename T, std::enable_if_t<std::is_long_double_v<T>, int> = 0>
static Type *getLLVMTypeImpl(LLVMContext &ctx) {
  llvm_unreachable("NOT IMPLEMENTED: getTypeFor<long double>()");
}

template <typename T, std::enable_if_t<!std::is_pointer_v<T>, int>>
Type *llvm::getLLVMTypeFor(LLVMContext &ctx) {
  return getLLVMTypeImpl<T>(ctx);
}

template Type *llvm::getLLVMTypeFor<bool>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<int8_t>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<uint8_t>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<int16_t>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<uint16_t>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<int32_t>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<uint32_t>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<int64_t>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<uint64_t>(LLVMContext &);

template Type *llvm::getLLVMTypeFor<const bool>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<const int8_t>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<const uint8_t>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<const int16_t>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<const uint16_t>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<const int32_t>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<const uint32_t>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<const int64_t>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<const uint64_t>(LLVMContext &);

template Type *llvm::getLLVMTypeFor<float>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<const float>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<double>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<const double>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<long double>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<const long double>(LLVMContext &);

template Type *llvm::getLLVMTypeFor<char>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<const char>(LLVMContext &);

// On some systems, long long and int64_t are the same type where explicitly
// instantiating both results in an error. __unix__ below implies __linux__
// and *BSD, but not MacOSX
#if __unix__
template Type *llvm::getLLVMTypeFor<long long>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<unsigned long long>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<const long long>(LLVMContext &);
template Type *llvm::getLLVMTypeFor<const unsigned long long>(LLVMContext &);
#endif // __unix__
