//===- Instrumentation.h - Utilities to support instrumentation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to support instrumentation of code. The utilities here can be used
// for both printf-style debugging of passes, and broader instrumentation of
// code.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_INSTRUMENTATION_H
#define KITSUNE_CORE_INSTRUMENTATION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"

namespace llvm {

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

} // namespace llvm

#endif // KITSUNE_CORE_INSTRUMENTATION_H
