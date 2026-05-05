//==- CGKitsune.h - Utilities for Kitsune-specific codegen -*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for Kitsune-specific LLVM IR generation (i.e. "clang codegen").
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGKITSUNE_H
#define LLVM_CLANG_LIB_CODEGEN_CGKITSUNE_H

#include "CGCall.h"
#include "CGValue.h"
#include "CodeGenFunction.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/ArrayRef.h"

namespace clang {

class Attr;

namespace CodeGen {

/// Get the value of the tapir::target attribute if it was set. If the
/// attribute was not set, return the primary tapir target \p primaryTT. This
/// will typically be the value of the --tapir command-line option.
llvm::TTID getTTID(llvm::ArrayRef<const Attr *> attrs, llvm::TTID primaryTT);

/// Get the value of the kitsune::launch attribute if it was set. If the
/// attribute was not set, return \p tpb.
unsigned getLaunchTPB(llvm::ArrayRef<const Attr *> attrs, unsigned tpb);

/// Is \p id the ID for a Kitsune-specific builtin.
bool IsKitBuiltin(unsigned builtinID);

/// Emit a call to a Kitsune-specific builtin, \p id. This should only be called
/// with the ID of a Kitsune-specific builtin.
RValue EmitKitBuiltinCall(CodeGenFunction &cgf, const FunctionDecl *funcDecl,
                          unsigned builtinID, const CallExpr *theCallExpr,
                          ReturnValueSlot rv);

/// Set kitsune-specific attributes on an LLVM global variable. This applies
/// to both declarations and definitions.
void AddKitAttributes(CodeGenModule &cgm, const VarDecl &vd,
                      llvm::GlobalVariable &g);

/// Set kitsune-specific attributes on an LLVM function and/or its arguments.
/// This applies to both declarations and definitions.
void AddKitAttributes(CodeGenModule &cgm, const FunctionDecl &fd,
                      llvm::Function &f);

} // namespace CodeGen

} // namespace clang

#endif // LLVM_CLANG_LIB_CODEGEN_CGKITSUNE_H
