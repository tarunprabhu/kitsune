//=- GenerateKitsuneCtorsImpl.h - Private header for ctor gen -----*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generate Kitsune global constructors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GENERATE_KITSUNE_CTORS_IMPL_H
#define LLVM_GENERATE_KITSUNE_CTORS_IMPL_H

#include <functional>

#define DEBUG_TYPE "generate-kitsune-ctors"

namespace llvm {

class Function;
class Module;
class TapirTargetOptions;
class TargetLibraryInfo;

namespace detail {

using GetTLI = std::function<TargetLibraryInfo &(Function &)>;

/// Generate a ctor for kitsune's cuda runtime.
void genKitsuneCtorCuda(Module &m, const TapirTargetOptions &tto,
                        GetTLI getTLI);

/// Generate a ctor for kitsune's hip runtime.
void genKitsuneCtorHip(Module &m, const TapirTargetOptions &tto, GetTLI getTLI);

} // namespace detail

} // namespace llvm

#endif // LLVM_GENERATE_KITSUNE CTORS_IMPL_H
