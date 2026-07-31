//===- PrintfDebugging.h - Utilities for printf-debugging -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to support "printf-debugging" of Kitsune's transformation passes.
// These make it easy to add printf calls to LLVM-IR. They are intended to be
// useful during development of passes - it is unlikely that one will ever find
// uses of these utilities in code that is not being actively developed or
// debugged.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_PRINTF_DEBUGGING_H
#define KITSUNE_CORE_PRINTF_DEBUGGING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"

namespace llvm {

class Value;

/// \addtogroup kitsune
/// @{

/// Insert a call to fprintf to write to stdout. The call will be inserted at
/// the insertion point of the given IRBuilder. \p args are passed as is to
/// printf. Their types must be consistent with the contents of the format
/// string \p fmt. Returns the inserted call.
///
/// The insertion point of the builder must be valid.
Value *insertPrintStdout(IRBuilder<> &builder, StringRef fmt,
                         ArrayRef<Value *> args, StringRef name = "");

/// Insert a call to fprintf to write to stdout. The call will be inserted
/// before the insertion point \p insertPt. \p args are passed as is to printf.
/// Their types must be consistent with the contents of the format string
/// \p fmt. Returns the inserted call.
///
/// \p insertPt must be valid.
Value *insertPrintStdout(InsertPosition insertPt, StringRef fmt,
                         ArrayRef<Value *> args, StringRef name = "");

/// Insert a call to fprintf to write to stderr. The call will be inserted at
/// the insertion point of the given IRBuilder. \p args are passed as is to
/// printf. Their types must be consistent with the contents of the format
/// string \p fmt. Returns the inserted call.
///
/// The insertion point of the builder must be valid.
Value *insertPrintStderr(IRBuilder<> &builder, StringRef fmt,
                         ArrayRef<Value *> args, StringRef name = "");

/// Insert a call to fprintf to write to stderr. The call will be inserted
/// before the insertion point \p insertPt. \p args are passed as is to printf.
/// Their types must be consistent with the contents of the format string
/// \p fmt. Returns the inserted call.
///
/// \p insertPt must be valid.
Value *insertPrintStderr(InsertPosition insertPt, StringRef fmt,
                         ArrayRef<Value *> args, StringRef name = "");

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_PRINTF_DEBUGGING_H
