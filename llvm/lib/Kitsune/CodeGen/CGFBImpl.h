//=- CGFB.h - Private header for codegen fat binaries pass --------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generate fat binaries from embedded bitcode
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_KITSUNE_CGFB_IMPL_H
#define LLVM_KITSUNE_CGFB_IMPL_H

#define DEBUG_TYPE "kit-cgfb"

namespace llvm {

class GlobalVariable;
class TapirTargetOptions;
class ToolOutputFile;

namespace detail {

/// Options to control code generation of the fat binaries. These are specific
/// to the way the kit-cgfb pass operates. Target-specific options will be set
/// in LLVM's \ref TargetOptions object from the \ref TapirTargetOptions.
struct CGFBOptions {
  /// Don't delete any intermediate files that were generated in the course of
  /// fat binary generation.
  bool keepFiles = false;

  /// Print the command lines for any external tools that are called during
  /// fat binary generation. This is only really useful for debugging.
  bool printCommandLines = false;
};

/// Copy the bytes in the given file containing a fat binary containing into the
/// initializer of the given global variable.
void embedFatBinary(ToolOutputFile &fatbinFile, GlobalVariable &g);

/// Codegen a fat binary for NVIDIA GPU's. \ref gfb is the global variable into
/// which the fat binary will be saved. \ref gbc is the global variable
/// containing the bitcode from which the fat binary will be generated. If
/// \ref keepFiles is true, the intermediate files generated during the
/// compilation process will not be deleted after use.
bool cgfbCuda(GlobalVariable &gfb, const GlobalVariable &gbc,
              const TapirTargetOptions &tto, const CGFBOptions& cgfbOpts);

/// Codegen a fat binary for AMD GPU's. \ref gfb is the global variable into
/// which the fat binary will be saved. \ref gbc is the global variable
/// containing the bitcode from which the fat binary will be generated. If
/// \ref keepFiles is true, the intermediate files generated during the
/// compilation process will not be deleted after use.
bool cgfbHip(GlobalVariable &gfb, const GlobalVariable &gbc,
             const TapirTargetOptions &tto, const CGFBOptions& cgfbOpts);

} // namespace detail

} // namespace llvm

#endif // LLVM_KITSUNE_CGFB_IMPL_H
