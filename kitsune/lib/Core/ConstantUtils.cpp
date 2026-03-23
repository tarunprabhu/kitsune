//===- ConstantUtils.cpp - Helper functions for constants -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper functions for constants.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ConstantUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

using namespace llvm;

GlobalVariable *llvm::createConstString(StringRef s, Module &m,
                                        StringRef name) {
  for (GlobalVariable &g : m.globals())
    if (g.isConstant() and g.hasInitializer())
      if (auto *cda = dyn_cast<ConstantDataArray>(g.getInitializer()))
        if (cda->isCString() and cda->getAsCString() == s)
          return &g;

  LLVMContext &ctx = m.getContext();
  Constant *init = ConstantDataArray::getString(ctx, s, true);
  Type *type = init->getType();
  auto *g = new GlobalVariable(m, type, /*IsConstant=*/true,
                               GlobalValue::PrivateLinkage, init, name);
  g->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  g->setAlignment(Align(1));

  return g;
}

Constant *llvm::stripCasts(Constant *c) {
  if (auto *cst = dyn_cast_or_null<ConstantExpr>(c))
    if (cst->isCast())
      return stripCasts(cst->getOperand(0));
  return c;
}

const Constant *llvm::stripCasts(const Constant *c) {
  if (const auto *cst = dyn_cast_or_null<ConstantExpr>(c))
    if (cst->isCast())
      return stripCasts((const Constant *)cst->getOperand(0));
  return c;
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, int>>
Constant *llvm::toConstant(const T &val, LLVMContext &ctx) {
  return ConstantInt::get(getLLVMTypeFor<T>(ctx), val);
}
template Constant *llvm::toConstant(const int8_t &val, LLVMContext &ctx);
template Constant *llvm::toConstant(const uint8_t &val, LLVMContext &ctx);
template Constant *llvm::toConstant(const int16_t &val, LLVMContext &ctx);
template Constant *llvm::toConstant(const uint16_t &val, LLVMContext &ctx);
template Constant *llvm::toConstant(const int32_t &val, LLVMContext &ctx);
template Constant *llvm::toConstant(const uint32_t &val, LLVMContext &ctx);
template Constant *llvm::toConstant(const int64_t &val, LLVMContext &ctx);
template Constant *llvm::toConstant(const uint64_t &val, LLVMContext &ctx);

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int>>
Constant *llvm::toConstant(const T &val, LLVMContext &ctx) {
  return ConstantFP::get(getLLVMTypeFor<T>(ctx), val);
}
template Constant *llvm::toConstant(const float &val, LLVMContext &ctx);
template Constant *llvm::toConstant(const double &val, LLVMContext &ctx);

template <typename T, std::enable_if_t<std::is_same_v<T, StringRef> ||
                                           std::is_same_v<T, StringLiteral> ||
                                           std::is_same_v<T, std::string>,
                                       int>>
Constant *llvm::toConstant(const T &val, LLVMContext &ctx) {
  return ConstantDataArray::getString(ctx, val, /*AddNull=*/false);
}
template Constant *llvm::toConstant(const StringLiteral &val, LLVMContext &ctx);
template Constant *llvm::toConstant(const StringRef &val, LLVMContext &ctx);
template Constant *llvm::toConstant(const std::string &val, LLVMContext &ctx);

template <typename T, std::enable_if_t<std::is_integral_v<T>, int>>
std::optional<T> llvm::fromConstant(const Constant &c) {
  if (const auto *cint = dyn_cast<ConstantInt>(&c))
    if (cint->getBitWidth() == sizeof(T) * 8)
      return cint->getLimitedValue();
  return std::nullopt;
}
template std::optional<int8_t> llvm::fromConstant(const Constant &c);
template std::optional<uint8_t> llvm::fromConstant(const Constant &c);
template std::optional<int16_t> llvm::fromConstant(const Constant &c);
template std::optional<uint16_t> llvm::fromConstant(const Constant &c);
template std::optional<int32_t> llvm::fromConstant(const Constant &c);
template std::optional<uint32_t> llvm::fromConstant(const Constant &c);
template std::optional<int64_t> llvm::fromConstant(const Constant &c);
template std::optional<uint64_t> llvm::fromConstant(const Constant &c);

template <typename T,
          std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float>, int>>
std::optional<T> llvm::fromConstant(const Constant &c) {
  if (const auto *cfp = dyn_cast<ConstantFP>(&c)) {
    const APFloat &apf = cfp->getValue();
    const fltSemantics &semantics = apf.getSemantics();
    if (APFloat::semanticsSizeInBits(semantics) == sizeof(T) * 8)
      return apf.convertToFloat();
  }
  return std::nullopt;
}
template std::optional<float> llvm::fromConstant(const Constant &c);

template <typename T,
          std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, double>, int>>
std::optional<T> llvm::fromConstant(const Constant &c) {
  if (const auto *cfp = dyn_cast<ConstantFP>(&c)) {
    const APFloat &apf = cfp->getValue();
    const fltSemantics &semantics = apf.getSemantics();
    if (APFloat::semanticsSizeInBits(semantics) == sizeof(T) * 8)
      return apf.convertToDouble();
  }
  return std::nullopt;
}
template std::optional<double> llvm::fromConstant(const Constant &c);

template <typename T, std::enable_if_t<std::is_same_v<T, StringRef>, int>>
std::optional<T> llvm::fromConstant(const Constant &c) {
  if (const auto *cda = dyn_cast<ConstantDataArray>(&c)) {
    if (cda->isString())
      return cda->getAsString();
    else if (cda->isCString())
      return cda->getAsCString();
  }
  return std::nullopt;
}
template std::optional<StringRef> llvm::fromConstant(const Constant &c);
