//===- HipABI.h - Tapir target for Kitsune's hip runtime --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
//  All rights reserved.
//
// Copyright 2021, 2023. Los Alamos National Security, LLC. This
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
// Tapir target that lowers to Kitsune's hip runtime
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TARGETS_HIP_ABI_H
#define KITSUNE_TARGETS_HIP_ABI_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

#include <set>

namespace llvm {

class TTOptions;

/// The tapir target to lower tapir loops to kitsune's hip runtime. The tapir
/// loops will be converted to GPU kernels.
/// \ingroup kitsune
class HipABI : public TapirTarget {
public:
  HipABI(Module &hostM, const TTOptions &tto);
  ~HipABI();

  /// Lower a call to the tapir.loop.grainsize intrinsic into a grain size
  /// (coarsening) value.
  Value *lowerGrainsizeCall(CallInst *grainsizeCall) override final;

  /// Lower the given Tapir sync instruction.
  void lowerSync(SyncInst &si) override final;

  void preProcessModule() override final;

  /// Process Function f before any function outlining is performed.  This
  /// routine should not modify the CFG structure.
  bool preProcessFunction(Function &f, TaskInfo &ti,
                          bool processingTapirLoops) override {
    return false;
  }

  // Add attributes to the Function helper produced from outlining a task.
  void addHelperAttributes(Function &helper) override {}

  // Pre-process the Function f that has just been outlined from a task.  This
  // routine is executed on each outlined function by traversing in post-order
  // the tasks in the original function.
  void preProcessOutlinedTask(Function &f, Instruction *detachPt,
                              Instruction *tfCreate, bool isSpawner,
                              BasicBlock *bb) override {}

  // Post-process the Function f that has just been outlined from a task.  This
  // routine is executed on each outlined function by traversing in post-order
  // the tasks in the original function.
  void postProcessOutlinedTask(Function &f, Instruction *detachPtr,
                               Instruction *tfCreate, bool isSpawner,
                               BasicBlock *tfEntry) override {}

  // Pre-process the root Function f as a function that can spawn subtasks.
  void preProcessRootSpawner(Function &f, BasicBlock *tfEntry) override {}

  // Post-process the root Function f as a function that can spawn subtasks.
  void postProcessRootSpawner(Function &f, BasicBlock *tfEntry) override {}

  // Process the invocation of a task for an outlined function.  This routine is
  // invoked after processSpawner once for each child subtask.
  void processSubTaskCall(TaskOutlineInfo &toi, DominatorTree &dt) override {}

  // Process Function f at the end of the lowering process.
  void postProcessFunction(Function &f, bool outliningTapirLoops) override {}

  // Process the host-side module at the end of lowering all functions within
  // the module.
  void postProcessModule() override final;

  // Process a generated helper Function f produced via outlining, at the end of
  // the lowering process.
  void postProcessHelper(Function &f) override {}

  // Return the loop outline processor associated with this target.
  LoopOutlineProcessor *
  getLoopOutlineProcessor(const TapirLoopInfo *tl) override final;

private:
  /// Currently, we create a single module into which all tapir loops to be
  /// run on an AMD GPU are outlined. A loop outline processor is created for
  /// each tapir loop which will add the outlined code into this module. This
  /// will eventually be compiled executable GPU code.
  Module kernelModule;

  /// When outlining tapir loops into the \ref KernelModule, we need to generate
  /// a name for the outlined function. This name must be unique. In the absence
  /// of debug information, the computed outlined function name consists of a
  /// fixed base with an integer suffix that is incremented for each tapir loop
  /// that is encountered.
  unsigned nextKernelID = 0;
};

} // namespace llvm

#endif // KITSUNE_TARGETS_HIP_ABI_H
