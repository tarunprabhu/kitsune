#ifndef TAPIR_UTILS_H_
#define TAPIR_UTILS_H_

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Tapir/TapirTypes.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm 
{
  Function *extractDetachBodyToFunction(DetachInst &Detach,
                                        DominatorTree &DT, 
                                        AssumptionCache &AC,
                                        CallInst **call = nullptr);
}
#endif
