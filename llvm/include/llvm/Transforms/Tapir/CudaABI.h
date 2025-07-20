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

#ifndef LLVM_TRANSFORMS_TAPIR_CUDA_ABI_H
#define LLVM_TRANSFORMS_TAPIR_CUDA_ABI_H

#include "kitsune/Core/ReachableGlobals.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {

class TapirTargetOptions;

/// The tapir target to lower tapir loops to kitsune's cuda runtime. The tapir
/// loops will be converted to GPU kernels.
class CudaABI final : public TapirTarget {
public:
  CudaABI(Module &HostM, const TapirTargetOptions &TTO);
  ~CudaABI() {}

  /// Lower a call to the tapir.loop.grainsize intrinsic into a grain size
  /// (coarsening) value. For GPU codes we currently limit this to a value of 1.
  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override;

  /// Lower the given Tapir sync instruction.
  /// This does nothing because we unconditionally sync immediately after the
  /// kernel launch call is generated. Any stray syncs will be cleaned up
  /// automatically by a later pass.
  void lowerSync(SyncInst &SI) override {}

  void addHelperAttributes(Function &F) override {}

  /// Process a host module before any lowering is performed.
  void preProcessModule() override;

  /// Process the host function before any function outlining is performed.
  bool preProcessFunction(Function &F, TaskInfo &TI,
                          bool OutliningTapirLoops) override {
    // Always returns false since this does not modify the CFG.
    return false;
  }

  /// Process the host function at the end of the lowering process.
  void postProcessFunction(Function &F, bool OutliningTapirLoops) override {}

  /// Process the generated helper function, produced via outlining, at the
  /// end of the lowering process.
  void postProcessHelper(Function &F) override {}

  /// Pre-process the function that has just been outlined from a task.
  void preProcessOutlinedTask(Function &F, Instruction *DetachPt,
                              Instruction *TaskFrameCreate, bool IsSpawner,
                              BasicBlock *TFEntry) override {}

  /// Post-process the function that has just been outlined from a task.
  void postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                               Instruction *TaskFrameCreate, bool IsSpawner,
                               BasicBlock *TFEntry) override {}

  /// Pre-process the root function as a function that can spawn subtasks.
  void preProcessRootSpawner(Function &F, BasicBlock *TFEntry) override {}

  /// Post-process the root function as a function that can spawn subtasks.
  void postProcessRootSpawner(Function &F, BasicBlock *TFEntry) override {}

  /// Process the invocation of a task for an outlined function.  This routine
  /// is invoked after processSpawner once for each child subtask.
  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) override {}

  void postProcessModule() override;

  LoopOutlineProcessor *
  getLoopOutlineProcessor(const TapirLoopInfo *TL) override;

private:
  /// Currently, we create a single "device" module into which all tapir loops
  /// are outlined. This is that module. The actual module is stashed in the
  /// host as embedded bitcode. Since this object may be created before the pass
  /// pipeline is even constructed, the embedded module will not be available in
  /// the constructor. Instead, this owning pointer will set to non-null in
  /// \ref preProcessModule() since that is the earliest point at which the
  /// embedded module is guaranteed to be present.
  std::unique_ptr<Module> DevM = nullptr;

  /// The total number of tapir loops that have been seen by this tapir target.
  /// The body of every tapir loop is outlined into a function in the device
  /// module. In the absence of debug information, this is used to determine a
  /// unique name for each of these functions.
  unsigned LoopsSeen = 0;
};

/// The loop outline process for transforming a Tapir parallel loop into a
/// cuda kernel function.
class CudaLoop : public LoopOutlineProcessor {
private:
  /// For GPU targets, the tapir loops are outlined into a separate module. This
  /// is that module.
  Module &KernelModule;

  /// The name of the kernel into which the loop is outlined.
  std::string KernelName;

  /// The "inputs" to the tapir loop that will eventually become arguments to
  /// the "kernel" function.
  SmallVector<Value *, 5> KernelArgs;

  /// The GlobalValue's used in the loop that is being outlined. This includes
  /// functions, global variables, aliases and ifunc's.
  ReachableGlobals UsedGlobals;

  // FIXME: There doesn't seem to be a compelling reason to have these be
  // class-level variables. These can be set in the module constructor which
  // avoids the need to call getOrInsert() per tapir-loop, but it is not clear
  // how much that is saving us. We might as well move these into the functions
  // where they are used since they are only used in postProcessOutline().

  // Intrinsics to determine the thread index.
  Function *CUThreadIdxX = nullptr, *CUThreadIdxY = nullptr,
           *CUThreadIdxZ = nullptr;

  // Intrinsics to determine the block index.
  Function *CUBlockIdxX = nullptr, *CUBlockIdxY = nullptr,
           *CUBlockIdxZ = nullptr;

  // Intrinsics to determine the block dimensions.
  Function *CUBlockDimX = nullptr, *CUBlockDimY = nullptr,
           *CUBlockDimZ = nullptr;

  // Cuda/PTX grid dimensions access.
  Function *CUGridDimX = nullptr, *CUGridDimY = nullptr, *CUGridDimZ = nullptr;

public:
  /// Create a loop outline processor.
  /// @param M           The host module
  /// @param DevM        The module into which the device code will be outlined
  /// @param KernelName  The name of the function in @ref DevM into which the
  ///                    tapir loop will be outlined
  /// @param TTO         The tapir target options
  CudaLoop(Module &M, Module &DevM, StringRef KernelName,
           const TapirTargetOptions &TTO);
  ~CudaLoop();

  void setupLoopOutlineArgs(Function &F, ValueSet &HelperArgs,
                            SmallVectorImpl<Value *> &HelperInputs,
                            ValueSet &InputSet,
                            const SmallVectorImpl<Value *> &LCArgs,
                            const SmallVectorImpl<Value *> &LCInputs,
                            const ValueSet &TLInputsFixed) override final;

  unsigned getIVArgIndex(const Function &F,
                         const ValueSet &Args) const override final;

  unsigned getLimitArgIndex(const Function &F,
                            const ValueSet &Args) const override final;

  void preProcessTapirLoop(TapirLoopInfo &TL, ValueToValueMapTy &VMap) override;
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                          ValueToValueMapTy &VMap) override final;
  void processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                               DominatorTree &DT) override final;
  void remapData(ValueToValueMapTy &VMap) override final;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_CUDA_ABI_H
