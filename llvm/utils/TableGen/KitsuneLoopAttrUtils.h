//=- KitsuneLoopAttrUtils.h - Utilities for Kitsune's loop attrs ---*-C++-*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities shared by Kitsune's loop attribute emitters
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_KITSUNE_LOOP_ATTR_UTILS_H
#define LLVM_TABLEGEN_KITSUNE_LOOP_ATTR_UTILS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class Record;
} // namespace llvm

// Get the base name of the attribute. Here are some examples of how the enum
// name of the attribute will be translated to the base name:
//
//    Target          target
//    Grainsize       grainsize
//    PerfectDepth    perfect.depth
//    Dim3            dim3
//
std::string getBaseName(const llvm::Record &attr);

// Get the string that will appear in metadata in LLVM-IR for the loop
// attribute with the given name.
std::string getIRName(const llvm::Record &attr);

// Return true if the attribute in the given record only applies to tapir loops,
// false otherwise.
bool isTapirLoopOnly(const llvm::Record &attr);

#endif // LLVM_TABLEGEN_KITSUNE_LOOP_ATTR_UTILS_H
