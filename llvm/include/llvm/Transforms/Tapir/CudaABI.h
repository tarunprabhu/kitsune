

#ifndef CUDA_ABI_H_
#define CUDA_ABI_H_

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Tapir/LoopSpawning.h"
//#include "llvm/Transforms/Tapir/TapirUtils.h"


namespace llvm {

  class CudaABILoopSpawning : public LoopOutline {
    public:
      CudaABILoopSpawning(Loop *OrigLoop, 
                          ScalarEvolution &SE, 
                          LoopInfo *LI,
                          DominatorTree *DT,
                          AssumptionCache *AC, 
                          OptimizationRemarkEmitter &ORE) 
        : LoopOutline(OrigLoop, SE, LI, DT, AC, ORE)
      {}

      bool processLoop();

      virtual ~CudaABILoopSpawning() { /* no-op */ }

    private:
      uint32_t   nextKernelId = 0;
  };

  class CudaABI : public TapirTarget {
    public:
      CudaABI();
      Value *GetOrCreateWorker8(Function &F);
      void createSync(SyncInst &SI, 
                      ValueToValueMapTy &DetachCtxToStackFrame);
      Function *createDetach(DetachInst &Detach,
                             ValueToValueMapTy &DetachCtxToStackFrame,
                             DominatorTree &DT, 
                             AssumptionCache &AC);                      
      void preProcessFunction(Function &F);
      void postProcessFunction(Function &F);
      void postProcessingHelper(Function &F);
      bool processMain(Function &F);
  };

}

#endif 
