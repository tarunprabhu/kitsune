//===- MetadataUtils.cpp - Helper functions for LLVM's metadata -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of utilities for LLVM's metadata
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/MetadataUtils.h"
#include "kitsune/Core/Tapir.h"
#include "kitsune/Core/TypeUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"

using namespace llvm;

template <typename DstTy, typename SrcTy>
Metadata *llvm::makeTapirLoopMetadata(LLVMContext &ctx, StringRef name,
                                      const SrcTy &val) {
  if constexpr (std::is_integral_v<DstTy>) {
    Type *type = getLLVMTypeFor<DstTy>(ctx);
    Metadata *ops[] = {
        MDString::get(ctx, name),
        ConstantAsMetadata::get(ConstantInt::get(type, (DstTy)val))};
    return MDNode::get(ctx, ops);
  }
  llvm_unreachable("makeTapirLoopMetadata: Non-integral tapir loop metadata "
                   "is not supported");
}

template Metadata *
llvm::makeTapirLoopMetadata<uint32_t>(LLVMContext &, StringRef, const TTID &);
template Metadata *
llvm::makeTapirLoopMetadata<uint32_t>(LLVMContext &, StringRef,
                                      const TapirSpawnStrategy &);
template Metadata *llvm::makeTapirLoopMetadata<uint32_t>(LLVMContext &,
                                                         StringRef,
                                                         const uint32_t &);
