//===- CudaABI.h - Tapir target for Kitsune's cuda runtime ------*- C++ -*-===//
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
//===----------------------------------------------------------------------===//
//
// Tapir target that lowers to Kitsune's cuda runtime
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TARGETS_CUDA_ABI_H
#define KITSUNE_TARGETS_CUDA_ABI_H

#include "llvm/Transforms/Tapir/LoweringUtils.h"

#include <set>

namespace llvm {

class TTOptions;

/// The tapir target to lower tapir loops to kitsune's cuda runtime. The tapir
/// loops will be converted to GPU kernels.
/// \ingroup kitsune
class CudaABI : public TapirTarget {
public:
  CudaABI(Module &hostM, const TTOptions &tto);
  ~CudaABI();

  Value *lowerGrainsizeCall(CallInst *grainsizeCall) override;
  void lowerSync(SyncInst &si) override;

  void addHelperAttributes(Function &f) override {}
  void preProcessModule() override;
  bool preProcessFunction(Function &f, TaskInfo &ti,
                          bool outliningTapirLoops) override {
    return false;
  }
  void postProcessFunction(Function &f, bool outliningTapirLoops) override {}
  void postProcessHelper(Function &f) override {}

  void preProcessOutlinedTask(Function &f, Instruction *detachPt,
                              Instruction *taskFrameCreate, bool isSpawner,
                              BasicBlock *tfEntry) override {}

  void postProcessOutlinedTask(Function &f, Instruction *detachPt,
                               Instruction *taskFrameCreate, bool isSpawner,
                               BasicBlock *tfEntry) override {}

  void preProcessRootSpawner(Function &f, BasicBlock *tfEntry) override {}
  void postProcessRootSpawner(Function &f, BasicBlock *tfEntry) override {}

  void processSubTaskCall(TaskOutlineInfo &toi, DominatorTree &dt) override {}

  void postProcessModule() override;

  LoopOutlineProcessor *
  getLoopOutlineProcessor(const TapirLoopInfo *tl) override;

private:
  /// Currently, we create a single module into which all tapir loops to be
  /// run on an NVIDIA GPU are outlined. A loop outline processor is created for
  /// each tapir loop which will add the outlined code into this module. This
  /// will eventually be converted to PTX and from there to executable GPU code.
  Module kernelModule;

  /// When outlining tapir loops into the \ref KernelModule, we need to generate
  /// a name for the outlined function. This name must be unique. In the absence
  /// of debug information, the computed outlined function name consists of a
  /// fixed base with an integer suffix that is incremented for each tapir loop
  /// that is encountered.
  unsigned nextKernelID = 0;
};

} // namespace llvm

#endif // KITSUNE_TARGETS_CUDA_ABI_H
