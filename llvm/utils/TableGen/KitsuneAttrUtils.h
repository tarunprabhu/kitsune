//===- KitsuneAttrUtils.h - Utilities for Kitsune's attributes ---*-C++-*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities shared by Kitsune's attribute emitters.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_KITSUNE_ATTR_UTILS_H
#define LLVM_TABLEGEN_KITSUNE_ATTR_UTILS_H

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
std::string getAttrBaseName(const llvm::Record &attr);

// Get the string that will appear in LLVM-IR for an instruction attribute.
std::string getInstAttrIRName(const llvm::Record &attr);

// Get the string that will appear in LLVM-IR for a loop attribute.
std::string getLoopAttrIRName(const llvm::Record &attr);

// Get the string that will appear in LLVM-IR metadata for a module attribute.
std::string getModuleAttrIRName(const llvm::Record &attr);

// Return true if the attribute in the given record only applies to tapir loops,
// false otherwise.
bool isTapirLoopOnly(const llvm::Record &attr);

#endif // LLVM_TABLEGEN_KITSUNE_ATTR_UTILS_H
