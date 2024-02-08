//===- OptimizationLevel.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/OptimizationLevel.h"

using namespace llvm;

// The definitions of the OptimizationLevel's have been moved to
// llvm/lib/Frontend/Tapir/OptLevelUtils.cpp (which is part of the
// libLLVMFrontendTapir.a static library). TapirTargetOptions, defined in
// llvm/lib/Frontend/Tapir requires OptimizationLevel which introduces a
// dependence on Passes. However, this results in a circular dependence because
// Passes => TapirOpts => FrontendTapir => Passes. Since Kitsune uses this
// OptimizationLevel object extensively in the frontend as well as in its own
// passses (unlike clang, flang and LLVM), what we really need to do is it to
// move this OptimizationLevel to a library that is shared by all Kitsune
// frontends and backends. This requires more careful thought about the design
// because it is far too easy to introduce circular dependences like this.
//
// For now, we do a rather horrible thing by leaving the declaration of
// OptimizationLevel in llvm/include/llvm/Passes/OptimizationLevel (to avoid
// changes in several parts of clang, flang and LLVM), but move the definition
// of the OptimizationLevel static class members to LLVMFrontendTapir. This is
// truly awful because it breaks all sorts of code organization conventions in
// LLVM - not to mention could result in unexpected missing/duplicate definition
// errors if tools attempt to use libLLVMPasses.a. However, we don't really care
// about the latter case, and, for now at least, LLVM builds with
// LLVM_LINK_LLVM_DYLIB=0 and LLVM_LINK_LLVM_DYLIB=1. What has not been tested
// is BUILD_SHARED_LIBS=1 which will probably break as a result of this change,
// so we choose to disable support for that configure-time option for now.
//
// 2025-May-05.
