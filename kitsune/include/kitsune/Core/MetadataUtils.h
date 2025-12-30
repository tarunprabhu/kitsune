//===- MetadataUtils.h - Helper functions for LLVM's metadata --*- C++ -*--===//
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

#ifndef KITSUNE_CORE_METADATA_UTILS_H
#define KITSUNE_CORE_METADATA_UTILS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

class LLVMContext;
class Metadata;

/// Construct an MDNode for tapir loop metadata.
///
/// \tparam DstTy The C++ type of the value in the metadata node. This should be
/// a C++ type, e.g. uint32_t if the \tparam SrcType should be appear as a
/// 32-bit integer in the constructed metadata node
/// \tparam SrcTy The type of the value to be serialized in the loop metadata
/// node. This is the type of \p val.
/// \param ctx The LLVM context object.
/// \param name A string to use as the first operand of the MDNode that will be
/// constructed. This will typically be of the form tapir.loop.*
/// \param val The value to add as the second operand of the MDNode to be
/// constructed. This is the actual value of the tapir loop metadata whose name
/// is \p name.
template <typename DstTy, typename SrcTy>
Metadata *makeTapirLoopMetadata(LLVMContext &ctx, StringRef name,
                                const SrcTy &val);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_METADATA_UTILS_H
