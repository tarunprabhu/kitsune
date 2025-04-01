/**
 ***************************************************************************
 * TODO: Need to update LANL/Triad Copyright notice...
 *
 * Copyright (c) 2017, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 *
 ***************************************************************************/

#ifndef LLVM_CLANG_BASIC_KITSUNE_OPTIONS_H
#define LLVM_CLANG_BASIC_KITSUNE_OPTIONS_H

#include "kitsune/Config/config.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

#include <vector>

// FIXME: We should find some other place to put this file.
//
// The KitsuneOptions object has to be shared between clang and flang. flang is
// actively attempting to remove all dependences on clang except the driver, so
// leaving this in clang/Basic (where it is currently) is not desirable.
// Currently, there is no other reasonable place to put it, so we keep this file
// "self-contained" by implementing everything right here. This way, it can be
// used by both clang and flang and doesn't complicate the build system by
// introducing new dependencies. But we should find a better way of doing this.

namespace clang {

/// Options that are Kitsune-specific. These affect both the Kitsune "language"
/// i.e. forall, spawn, sync etc. and the backend code-generation via Tapir.
class KitsuneOptions {
private:
  /// Is a Kitsune frontend being used. The frontend could be used without a
  /// tapir target, so we can't use the @ref TapirTarget field to determine
  /// whether we are using Kitsune.
  unsigned kitsuneFrontend : 1;

  /// Is "Kokkos mode" enabled.
  unsigned kokkos : 1;

  /// If "Kokkos mode" is enabled, should the initialization of libkokkoscore
  /// be overrident.
  unsigned kokkosNoInit : 1;

  /// Should loop strip-mining be enabled.
  unsigned stripmineLoops : 1;

  /// Enable verbose mode for the tapir target LLVM passes. This is different
  /// from -mllvm -debug-only=<tapir-target-llvm-pass-name>. When the verbose
  /// flag is set, the passes may print some subset of the information that is
  /// printed in full debug mode. However, there is no guarantee that any given
  /// tapir target LLVM pass will print anything at all.
  unsigned tapirTargetVerbose : 1;

  /// If true, code will be generated to enable verbose mode in the Kitsune
  /// runtime. This obviates the need to set an environment variable to turn on
  /// verbose mode when the compiled code is run. The effect of this will only
  /// be visible in certain tapir targets because not all use Kitsune's runtime.
  unsigned kitsuneRuntimeVerbose : 1;

  /// The TapirTarget to enable for code generation.
  ///
  /// For now, this is optional because we do not have a default tapir target
  /// and even when using the Kitsune frontends (kitcc, kit++ etc.), a flag with
  /// a Tapir target must be provided to enable the use of the Tapir IR
  /// constructs. If this field is set to some "non-empty" value, it implies
  /// that the Kitsune "language" mode has been enabled.
  ///
  /// This may have to be changed in order to handle the "inline" Tapir
  /// attributes including those needed for multi-target support.
  ///
  std::optional<llvm::TapirTargetID> tapirTarget = std::nullopt;

  /// Path to the OpenCilk ABI bitcode file. This will only be non-empty if the
  /// OpenCilk tapir target is enabled.
  std::optional<std::string> opencilkABIBitcodeFile = std::nullopt;

  /// If this is non-zero, the number of threads per block to use.
  unsigned fixedThreadsPerBlock = 0;

  /// If this is non-zero, the maximum number of threads per block to use. This
  /// may be used in conjunction with @ref threadsPerBlock, in which case this
  /// value must be greater than or equal to @ref threadsPerBlock.
  unsigned maxThreadsPerBlock = 0;

  /// The NVIDIA GPU architecture for which to generate code. This is only
  /// relevant for the cuda tapir target, although the default is always set.
  /// This is a string and not an enum because it is not clear if anything is to
  /// be gained by making it an enum. So far, all uses of this are as a string.
  std::string cudaArch = KITSUNE_CUDA_ARCH_DEFAULT;

  /// The AMD GPU architecture for which to generate code. This is only relevant
  /// for the hip tapir target, although the default is always set.
  /// This is a string and not an enum because it is not clear if anything is to
  /// be gained by making it an enum. So far, all uses of this are as a string.
  std::string hipArch = KITSUNE_HIP_ARCH_DEFAULT;

public:
  void setKitsuneFrontend(bool kitsuneFrontend = true) {
    this->kitsuneFrontend = kitsuneFrontend;
  }

  void setKokkos(bool kokkos = true) { this->kokkos = kokkos; }

  void setKokkosNoInit(bool kokkosNoInit = true) {
    this->kokkosNoInit = kokkosNoInit;
  }

  void setStripmineLoops(bool stripmineLoops = true) {
    this->stripmineLoops = stripmineLoops;
  }

  void setKitsuneRuntimeVerbose(bool verbose = true) {
    this->kitsuneRuntimeVerbose = verbose;
  }

  void setTapirTargetVerbose(bool verbose = true) {
    this->tapirTargetVerbose = verbose;
  }

  void setTapirTarget(llvm::TapirTargetID tapirTarget) {
    this->tapirTarget = tapirTarget;
  }

  void setOpenCilkABIBitcodeFile(llvm::StringRef file) {
    this->opencilkABIBitcodeFile = file;
  }

  void setCudaArch(llvm::StringRef arch) {
    this->cudaArch = arch;
  }

  void setHipArch(llvm::StringRef arch) {
    this->hipArch = arch;
  }

  void setFixedThreadsPerBlock(unsigned threadsPerBlock) {
    this->fixedThreadsPerBlock = threadsPerBlock;
  }

  void setMaxThreadsPerBlock(unsigned threadsPerBlock) {
    this->maxThreadsPerBlock = threadsPerBlock;
  }

  bool isKitsuneFrontend() const { return kitsuneFrontend; }

  bool hasTapirTarget() const { return tapirTarget.has_value(); }

  bool getKokkos() const { return kokkos; }

  bool getKokkosNoInit() const { return kokkosNoInit; }

  bool getStripmineLoops() const { return stripmineLoops; }

  bool getTapirTargetVerbose() const { return tapirTargetVerbose; }

  bool getKitsuneRuntimeVerbose() const { return kitsuneRuntimeVerbose; }

  /// This should only be called when a TapirTarget is known to exist.
  std::optional<llvm::TapirTargetID> getTapirTarget() const {
    return tapirTarget;
  }

  std::optional<llvm::StringRef> getOpenCilkABIBitcodeFile() const {
    return opencilkABIBitcodeFile;
  }

  unsigned getFixedThreadsPerBlock() const { return fixedThreadsPerBlock; }

  unsigned getMaxThreadsPerBlock() const { return maxThreadsPerBlock; }

  llvm::StringRef getCudaArch() const { return cudaArch; }

  llvm::StringRef getHipArch() const { return hipArch; }
};

} // namespace clang

#endif // LLVM_CLANG_BASIC_KITSUNE_OPTIONS_H
