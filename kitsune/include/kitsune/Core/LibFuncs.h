//===- LibFuncs.h - Utilities for Kitsune's library functions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to work with Kitsune's library functions. These are mainly useful
// when lowering Kitsune's intrinsics, but they can be used whenever these
// library functions need to be used directly in IR.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_LIBFUNCS_H
#define KITSUNE_CORE_LIBFUNCS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"

namespace llvm {

class FunctionType;
class LLVMContext;
class Module;

/// \addtogroup kitsune
/// \@{

/// Kitsune's known library functions. These are those functions declared in
/// kitsune/Core/LibFuncs.td. Generally, any function that is exposed by
/// libkitrt should be present here.
enum class KitFunc {
#define GET_LIBFUNC_ENUMS
#include "kitsune/Core/LibFuncs.inc"
};

/// Get the name of the given library function.
StringRef getLibFuncName(KitFunc libFunc);

/// Get the type of the given library function.
FunctionType *getLibFuncType(KitFunc libFunc, LLVMContext &ctx);

/// Get or insert a declaration for the given library function.
FunctionCallee getOrInsertLibFunc(Module &m, KitFunc libFunc);

/// \@}

} // namespace llvm

#endif // KITSUNE_CORE_LIBFUNCS_H
