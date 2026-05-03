//===- CGKitsuneUtils.h - Utilities used in Kitsune's codegen ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities used in Kitsune's clang CodeGen
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGKITSUNEUTILS_H
#define LLVM_CLANG_LIB_CODEGEN_CGKITSUNEUTILS_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/ArrayRef.h"

namespace clang {

class Attr;

namespace CodeGen {

/// Get the value of the tapir::target attribute if it was set. If the
/// attribute was not set, return \p defawlt.
llvm::TTID getTapirTarget(llvm::ArrayRef<const Attr *> attrs,
                          llvm::TTID defawlt);

/// Get the value of the kitsune::launch attribute if it was set. If the
/// attribute was not set, return 0.
unsigned getKitsuneLaunchAttr(llvm::ArrayRef<const Attr *> attrs);

} // namespace CodeGen

} // namespace clang

#endif // LLVM_CLANG_LIB_CODEGEN_CGKITSUNEUTILS_H
