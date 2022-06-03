//===- CudaABI.h - Interface to the Kitsune Cuda back end -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the Tapir to CUDA transformation that targets the direct CUDA 
// support in the Kitsune runtime API.  The transform currently targets 
// ahead-of-time (non-JIT) code generation and should allow cross-compilation 
// on systems without a local GPU -- as long as the CUDA toolchain is 
// available. 
// 
//===----------------------------------------------------------------------===//
#ifndef TapirCuda_ABI_H_
#define TapirCuda_ABI_H_

#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Tapir/LocalizeGlobals.h"

namespace llvm {

class TargetMachine;
class CudaLoop;

class CudaABI : public TapirTarget {

public:
  CudaABI(Module &M);
  ~CudaABI();

  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void lowerSync(SyncInst &SI) override final;

  void addHelperAttributes(Function &F) override final;
  void preProcessFunction(Function &F, TaskInfo &TI,
                          bool OutliningTapirLoops) override final;
  void postProcessFunction(Function &F, bool OutliningTapirLoops) override final;
  void postProcessHelper(Function &F) override final;

  void preProcessOutlinedTask(Function &F,
                              Instruction *DetachPt,
                              Instruction *TaskFrameCreate,
                              bool IsSpawner,
                              BasicBlock *TFEntry) override final;

  void postProcessOutlinedTask(Function &F,
                               Instruction *DetachPt,
                               Instruction *TaskFrameCreate,
                               bool IsSpawner,
                               BasicBlock *TFEntry) override final;

  void preProcessRootSpawner(Function &F,
                             BasicBlock *TFEntry) override final;
  void postProcessRootSpawner(Function &F,
                              BasicBlock *TFEntry) override final;

  void processSubTaskCall(TaskOutlineInfo &TOI,
                          DominatorTree &DT) override final;

  LoopOutlineProcessor *getLoopOutlineProcessor(const TapirLoopInfo *TL)
                                                const override final;
};

/// The loop outline process for transforming a Tapir parallel loop
/// represention into a Cuda runtime and PTX --> fat binary kernel
/// execution.
///
///  * The loop processor requires a CUDA install and that the 'ptxas'
///    and 'fatbinary' executables are in the user's path.  While it
///    is tempting to inline direct CUDA calls into the transform this
///    has two drawbacks:
///
///      1. CMake dependencies on Cuda would need to be added.
///      2. GPU would be required at compile time (i.e., no cross
///         compilation support).
///
class CudaLoop : public LoopOutlineProcessor {
  friend class CudaABI;

private:
  static unsigned NextKernelID;       // Give the generated kernel a unique ID.
  unsigned KernelID;                  // Unique ID for this transformed loop.
  std::string KernelName;             // A unique name for the kernel.
  Module  KernelModule;                // PTX module holds the generated kernel(s).
  TargetMachine  *PTXTargetMachine;

  // If the global variables used in the kernel are to be passed as explicitly
  // to the kernel as an additional parameter, this will carry out that
  // transformation.
  std::unique_ptr<LocalizeGlobals> localizeGlobals;

  bool Valid = false;

  FunctionCallee GetThreadIdx = nullptr;
  // Cuda/PTX thread index access.
  Function *CUThreadIdxX  = nullptr,
           *CUThreadIdxY  = nullptr,
           *CUThreadIdxZ  = nullptr;
  // Cuda/PTX block index and dimensions access.
  Function *CUBlockIdxX   = nullptr,
           *CUBlockIdxY   = nullptr,
           *CUBlockIdxZ   = nullptr;
  Function *CUBlockDimX   = nullptr,
           *CUBlockDimY   = nullptr,
           *CUBlockDimZ   = nullptr;
  // Cuda/PTX grid dimensions access.
  Function *CUGridDimX    = nullptr,
           *CUGridDimY    = nullptr,
           *CUGridDimZ    = nullptr;

  // Cuda thread synchronize
  Function *CUSyncThreads = nullptr;

  // Kitsune Cuda-centric runtime entry points.
  FunctionCallee KitCudaInitFn   = nullptr;
  FunctionCallee KitCudaLaunchFn = nullptr;
  FunctionCallee KitCudaWaitFn   = nullptr;
  FunctionCallee KitCudaMemPrefetchFn = nullptr;
  FunctionCallee KitCudaSetDefaultTBPFn = nullptr;
  FunctionCallee KitCudaSetDefaultLaunchParamsFn = nullptr;

  bool emitFatBinary();
  std::string createPTXFile();
  std::string createFatBinaryFile(const std::string &PTXFileName);

  SmallVector<Value *, 5> OrderedInputs;

public:
  CudaLoop(Module &M, 
           const std::string &KernelName, 
           bool MakeUniqueName = true);
  ~CudaLoop();

  void setupLoopOutlineArgs(Function &F, ValueSet &HelperArgs,
                            SmallVectorImpl<Value *> &HelperInputs,
                            ValueSet &InputSet,
                            const SmallVectorImpl<Value *> &LCArgs,
                            const SmallVectorImpl<Value *> &LCInputs,
                            const ValueSet &TLInputsFixed) override final;

  unsigned getIVArgIndex(const Function &F, const ValueSet &Args)
                         const override final;

  unsigned getLimitArgIndex(const Function &F, const ValueSet &Args)
                            const override final;

  std::string getKernelName() const { return KernelName; }
  void updateKernelName(const std::string &N, bool addID = false);

  void setValid(bool flag) { Valid = flag; }
  bool isValid() const { return Valid; }

  Constant * createConstantStr(const std::string &Str, 
                               const std::string &Name = "",
                               const std::string &SectionName = "",
                               unsigned Alignment = 0);

  unsigned getKernelID() const {
    return KernelID;
  }

  void transformForPTX();

  Constant *createKernelBuffer();
  Function *createCudaCtor(Constant *FatBinaryPtr);
  Function *createCudaDtor(GlobalVariable *BinHandle);

  void preProcessTapirLoop(TapirLoopInfo &TL, 
                           ValueToValueMapTy &VMap);

  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo & Out,
                          ValueToValueMapTy &VMap) override final;

  void processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo & TOI,
                               DominatorTree &DT) override final;
};
}

#endif
