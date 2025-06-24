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

/// Copy the bytes in the given file containing a fat binary containing into the
/// initializer of the given global variable.
void embedFatBinary(ToolOutputFile &fatbinFile, GlobalVariable &g);

/// Codegen a fat binary for NVIDIA GPU's. \ref gfb is the global variable into
/// which the fat binary will be saved. \ref gbc is the global variable
/// containing the bitcode from which the fat binary will be generated. If
/// \ref keepFiles is true, the intermediate files generated during the
/// compilation process will not be deleted after use.
bool cgfbCuda(GlobalVariable &gfb, const GlobalVariable &gbc,
              const TapirTargetOptions &tto, bool keepFiles);

/// Codegen a fat binary for AMD GPU's. \ref gfb is the global variable into
/// which the fat binary will be saved. \ref gbc is the global variable
/// containing the bitcode from which the fat binary will be generated. If
/// \ref keepFiles is true, the intermediate files generated during the
/// compilation process will not be deleted after use.
bool cgfbHip(GlobalVariable &gfb, const GlobalVariable &gbc,
             const TapirTargetOptions &tto, bool keepFiles);

} // namespace detail

} // namespace llvm

#endif // LLVM_KITSUNE_CGFB_IMPL_H
