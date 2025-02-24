//===- CudaABI.h - Tapir to the Kitsune runtime CUDA target -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
//
//  Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
//  All rights reserved.
//
//  Copyright 2021, 2023. Los Alamos National Security, LLC. This
//  software was produced under U.S. Government contract
//  DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which
//  is operated by Los Alamos National Security, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT
//  NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
//  OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
//  If software is modified to produce derivative works, such modified
//  software should be clearly marked, so as not to confuse it with
//  the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
// Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
//  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//===---------------------------------------------------------------------===//
//
#ifndef LLVM_TAPIR_GPU_ABI_OPTIONS_H
#define LLVM_TAPIR_GPU_ABI_OPTIONS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

namespace llvm {

/// Options common to all GPU tapir targets. The terminology used by NVIDIA and
/// AMD GPU's is sometimes different, though the underlying ideas are
/// effectively the same. In such cases, we use NVIDIA's terminology.
class GPUABIOptionsBase : public TapirTargetOptions {
protected:
  /// If true, enable verbose mode in this tapir target.
  unsigned Verbose : 1;

  /// If true, set the Kitsune runtime in verbose mode.
  unsigned RuntimeVerbose : 1;

  /// The optimization level set on the command line. This level will be used
  /// for both the middle-end optimizations on the kernel functions and the
  /// backend GPU code generators (including external assemblers as needed).
  llvm::OptimizationLevel OptLevel = llvm::OptimizationLevel::O0;

  /// The machine architecture for which to generate code.
  std::string Arch;

  /// How to use fuse floating-point operations.
  llvm::FPOpFusion::FPOpFusionMode FPOpFusionMode = llvm::FPOpFusion::Strict;

  /// If this is non-zero, this value will be used when launching all kernels
  /// which do not already have a custom FixedThreadsPerBlock value. If this is
  /// not set, the number of threads per block to use will be determined by the
  /// runtime.
  unsigned FixedThreadsPerBlock = 0;

  /// If this is non-zero, the threads per block will not be allowed to exceed
  /// this value.
  unsigned MaxThreadsPerBlock = 0;

protected:
  GPUABIOptionsBase(TapirTargetOptionsKind kind) : TapirTargetOptions(kind) {}

  void copyFrom(const GPUABIOptionsBase &src) {
    this->Verbose = src.Verbose;
    this->RuntimeVerbose = src.RuntimeVerbose;
    this->OptLevel = src.OptLevel;
    this->Arch = src.Arch;
    this->FPOpFusionMode = src.FPOpFusionMode;
    this->FixedThreadsPerBlock = src.FixedThreadsPerBlock;
    this->MaxThreadsPerBlock = src.MaxThreadsPerBlock;
  }

public:
  GPUABIOptionsBase(const GPUABIOptionsBase &) = delete;
  GPUABIOptionsBase &operator=(const GPUABIOptionsBase &) = delete;
  virtual ~GPUABIOptionsBase() = default;
  virtual GPUABIOptionsBase *clone() const override = 0;

  void setVerbose(bool verbose = true) { this->Verbose = verbose; }
  void setRuntimeVerbose(bool verbose = true) {
    this->RuntimeVerbose = verbose;
  }
  void setOptLevel(llvm::OptimizationLevel OptLevel) {
    this->OptLevel = OptLevel;
  }
  void setArch(llvm::StringRef Arch) { this->Arch = Arch; }
  void setFPOpFusionMode(llvm::FPOpFusion::FPOpFusionMode FPOpFusionMode) {
    this->FPOpFusionMode = FPOpFusionMode;
  }
  void setFixedThreadsPerBlock(unsigned ThreadsPerBlock) {
    this->FixedThreadsPerBlock = ThreadsPerBlock;
  }
  void setMaxThreadsPerBlock(unsigned ThreadsPerBlock) {
    this->MaxThreadsPerBlock = ThreadsPerBlock;
  }

  bool getVerbose() const { return Verbose; }
  bool getRuntimeVerbose() const { return RuntimeVerbose; }
  llvm::OptimizationLevel getOptLevel() const { return OptLevel; }
  llvm::StringRef getArch() const { return Arch; }
  llvm::FPOpFusion::FPOpFusionMode getFPOpFusionMode() const {
    return FPOpFusionMode;
  }
  unsigned getFixedThreadsPerBlock() const { return FixedThreadsPerBlock; }
  unsigned getMaxThreadsPerBlock() const { return MaxThreadsPerBlock; }

  static bool classof(const TapirTargetOptions *TTO) {
    return TTO->getKind() == TTO_Cuda || TTO->getKind() == TTO_Hip;
  }
};

} // namespace llvm

#endif // LLVM_TAPIR_GPU_ABI_OPTIONS_H
