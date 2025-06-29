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

#ifndef LLVM_KITSUNE_GENERATE_CTORS_IMPL_H
#define LLVM_KITSUNE_GENERATE_CTORS_IMPL_H

#include <functional>

#define DEBUG_TYPE "kit-ctors"

namespace llvm {

class Function;
class Module;
class TapirTargetOptions;
class TargetLibraryInfo;

namespace detail {

using GetTLI = std::function<TargetLibraryInfo &(Function &)>;

/// Options to generate global ctors for kitsune's runtime.
struct GenerateCtorOptions {
public:
  /// Enable refining the launch parameter in the runtime.
  unsigned refineLaunches : 1;

  /// Launch kernel using Y-axis threading.
  unsigned useYLaunch : 1;
};

/// Generate a ctor for kitsune's cuda runtime.
void genCtorCuda(Module &m, GetTLI getTLI, const TapirTargetOptions &tto,
                 const GenerateCtorOptions &ctorOpts);

/// Generate a ctor for kitsune's hip runtime.
void genCtorHip(Module &m, GetTLI getTLI, const TapirTargetOptions &tto,
                const GenerateCtorOptions &ctorOpts);

} // namespace detail

} // namespace llvm

#endif // LLVM_KITSUNE_GENERATE CTORS_IMPL_H
