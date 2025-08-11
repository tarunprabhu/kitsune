//===- SingletonUtils.h - Utilities for Kitsune's singletons ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kitsune uses some singletons to interface correctly with its own runtime, and
// the external runtimes used by some tapir targets (such as libcuda and
// libamdhip64). This provides utilities to work with these singletons.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_SINGLETON_UTILS_H
#define KITSUNE_CORE_SINGLETON_UTILS_H

#include "kitsune/Config/config.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class GlobalVariable;
class Module;

/// If the given string is the name of an embedded device code section, return
/// the tapir target that will have generated it.
std::optional<TTID> getTTIDForSection(StringRef section);

/// Get the name of the singleton global variable that will contain the fat
/// binary for the given tapir target.
StringLiteral getSingletonFBName(TTID tt);

/// Get the name of the section containing the singleton fat binary global
/// varible.
StringLiteral getSingletonFBSection(TTID tt);

/// Get the global variable created by a previous call to @ref
/// createSingletonFBGlobal with the given tapir target if one exists.
GlobalVariable *getSingletonFBGlobal(TTID tt, Module &m);

/// Create a global variable which will contains the fully linked fat binary.
/// This will have external linkage and no initializer since it will only become
/// available at link time.
///
/// @param tt The ID of the tapir target that is creating this embedded bitcode
/// @param m The host module into which the global variable will be created
/// @returns The newly created global variable
GlobalVariable *createSingletonFBGlobal(TTID tt, Module &m);

} // namespace llvm

#endif // KITSUNE_CORE_SINGLETON_UTILS_H
