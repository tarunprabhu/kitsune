//===- MetadataUtils.cpp - Utilities for LLVM's metadata ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's metadata.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/MetadataUtils.h"

using namespace llvm;

template <typename T, std::enable_if_t<std::is_string_like_v<T>, int>>
Metadata *llvm::toMetadata(const T &val, LLVMContext &ctx) {
  return MDString::get(ctx, val);
}
template Metadata *llvm::toMetadata(const StringLiteral &val, LLVMContext &ctx);
template Metadata *llvm::toMetadata(const StringRef &val, LLVMContext &ctx);
template Metadata *llvm::toMetadata(const std::string &val, LLVMContext &ctx);

template <
    typename T,
    std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>, int>>
Metadata *llvm::toMetadata(const T &val, LLVMContext &ctx) {
  return ConstantAsMetadata::get(toConstant(val, ctx));
}
template Metadata *llvm::toMetadata(const int8_t &val, LLVMContext &ctx);
template Metadata *llvm::toMetadata(const uint8_t &val, LLVMContext &ctx);
template Metadata *llvm::toMetadata(const int16_t &val, LLVMContext &ctx);
template Metadata *llvm::toMetadata(const uint16_t &val, LLVMContext &ctx);
template Metadata *llvm::toMetadata(const int32_t &val, LLVMContext &ctx);
template Metadata *llvm::toMetadata(const uint32_t &val, LLVMContext &ctx);
template Metadata *llvm::toMetadata(const int64_t &val, LLVMContext &ctx);
template Metadata *llvm::toMetadata(const uint64_t &val, LLVMContext &ctx);
template Metadata *llvm::toMetadata(const float &val, LLVMContext &ctx);
template Metadata *llvm::toMetadata(const double &val, LLVMContext &ctx);

template <typename T, std::enable_if_t<std::is_same_v<T, StringRef>, int>>
std::optional<T> llvm::fromMetadata(const Metadata *md) {
  if (auto *mdString = dyn_cast<MDString>(md))
    return mdString->getString();
  return std::nullopt;
}
template std::optional<StringRef> llvm::fromMetadata(const Metadata *md);

template <
    typename T,
    std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>, int>>
std::optional<T> llvm::fromMetadata(const Metadata *md) {
  if (auto *cmd = dyn_cast<ConstantAsMetadata>(md))
    if (auto *c = dyn_cast<Constant>(cmd->getValue()))
      return fromConstant<T>(*c);
  return std::nullopt;
}
template std::optional<int8_t> llvm::fromMetadata(const Metadata *md);
template std::optional<uint8_t> llvm::fromMetadata(const Metadata *md);
template std::optional<int16_t> llvm::fromMetadata(const Metadata *md);
template std::optional<uint16_t> llvm::fromMetadata(const Metadata *md);
template std::optional<int32_t> llvm::fromMetadata(const Metadata *md);
template std::optional<uint32_t> llvm::fromMetadata(const Metadata *md);
template std::optional<int64_t> llvm::fromMetadata(const Metadata *md);
template std::optional<uint64_t> llvm::fromMetadata(const Metadata *md);
template std::optional<float> llvm::fromMetadata(const Metadata *md);
template std::optional<double> llvm::fromMetadata(const Metadata *md);
