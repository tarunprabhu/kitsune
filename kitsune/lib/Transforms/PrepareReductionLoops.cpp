//===- PrepareReductionLoops.cpp - Transform tapir reduction loops --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform tapir loops that perform reductions to a form that is suitable for
// parallel execution. The actual transformations are carried out in different
// implementation files specialized for CPU and GPU. This defines utilities that
// are shared by those specializations.
//
//===----------------------------------------------------------------------===//

#include "PrepareReductionLoops.h"
#include "LoopWrapping.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/IRBuilderUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TTUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

#define DEBUG_TYPE "kit-prepare"

using namespace llvm;

// Collect the reduction intrinsics called in the loop being transformed,
// \p loop.
static SmallVector<ReductionInfo, 1> collectReductions(Loop &loop) {
  SmallVector<ReductionInfo, 1> reductions;

  for (BasicBlock *bb : loop.getBlocks())
    for (Instruction &inst : *bb)
      if (auto *call = dyn_cast<CallBase>(&inst))
        if (call->getIntrinsicID() == Intrinsic::kit_reduce_0)
          reductions.emplace_back(call);

  return reductions;
}

template <typename... Args>
static bool complain(const Loop &loop, DiagID diag, Args &&...args) {
  emitDiagnostic(loop, diag, args...);
  return false;
}

static bool prepareReductionLoopForSerialExecution(TapirLoopInfo &tapirLoop) {
  // There is nothing to be done to prepare a tapir reduction loop for the
  // serial tapir target. Calls to Kitsune's reduce intrinsics will be lowered
  // in a separate pass.
  addPreparedAttr(*tapirLoop.getLoop());
  return false;
}

AllocaInst *llvm::detail::createLocalResult(IRBuilder<> &builder,
                                            const ReductionInfo &redxn,
                                            bool initialize) {
  assert(builder.GetInsertBlock() &&
         "Insert point of builder must be a basic block");
  assert(getModule(builder) &&
         "Insert point of builder must be set to a basic block in a module");

  Value *value = redxn.value;
  Value *unit = redxn.unit;
  unsigned elemSize = redxn.elemSize;

  // We cannot, in general, get the type of the value being reduced from either
  // the type of the unit value, or the value being reduced. One or both of
  // these may be "passed by reference" - particularly in the case of custom
  // reductions on objects. In this case, we cannot know the type of the
  // underlying object being reduced since the pointers are opaque. Since the
  // element size is explicitly passed to this intrinsic, we alloca that many
  // bytes.
  LLVMContext &ctx = builder.getContext();
  Type *i8 = Type::getInt8Ty(ctx);
  ArrayType *resultTy = ArrayType::get(i8, elemSize);

  AllocaInst *result = builder.CreateAlloca(resultTy, nullptr, "reduc.partial");
  if (initialize) {
    Type *valueTy = value->getType();
    if (isa<PointerType>(valueTy)) {
      Align align = getTypeAlignment(*getModule(builder), valueTy);
      builder.CreateMemCpy(result, align, unit, align, elemSize);
    } else {
      builder.CreateStore(unit, result);
    }
  }

  return result;
}

bool llvm::detail::prepareReductionLoop(TapirLoopInfo &tapirLoop,
                                        DominatorTree &dt, LoopInfo &li,
                                        MemorySSA &mssa, ScalarEvolution &se,
                                        TaskInfo &ti) {
  auto sanityCheck = [](const Loop &loop) -> void {
    // These checks are mostly to ensure that the loop objects known to the
    // driver don't get corrupted when processing a function with many tapir
    // loops.
    assert(isTapirLoop(loop) &&
           "Loop with reduction attribute is a tapir loop");
    assert(hasReductionAttr(loop) &&
           "Loop is expected to have the reduction attribute");
    assert(!hasPreparedAttr(loop) &&
           "Tapir reduction loop has not been prepared");
  };

  Loop &loop = *tapirLoop.getLoop();
  sanityCheck(loop);

  // If the loop does not perform any actual reductions, mark it as prepared,
  // but return false because only the metadata will have changed.
  SmallVector<ReductionInfo, 1> reductions = collectReductions(loop);
  if (reductions.empty()) {
    addPreparedAttr(loop);
    return false;
  }

  TTID tt = *getTargetAttr(loop);
  if (tt == TTID::Serial)
    return prepareReductionLoopForSerialExecution(tapirLoop);
  else if (isCPUTT(tt))
    return detail::prepareReductionLoopForCPU(tapirLoop, reductions, dt, li,
                                              mssa, se, ti);
  else if (isGPUTT(tt))
    return detail::prepareReductionLoopForGPU(*tapirLoop.getLoop(), reductions,
                                              dt, li, mssa);
  llvm_unreachable("prepareReductionLoop: TT is neither CPU- nor GPU-centric");
}

bool llvm::detail::checkReductionLoop(TapirLoopInfo &tapirLoop,
                                      DominatorTree &dt, LoopInfo &li) {
  if (!checkTapirLoopSafeToWrap(tapirLoop, dt, li))
    return false;

  // If this is not a top-level tapir loop, it is nested within another tapir
  // loop.
  // FIXME: Add support for reductions in nested tapir loops.
  const Loop &loop = *tapirLoop.getLoop();
  if (!isTopLevelTapirLoop(loop))
    return complain(loop, DiagID::ErrGeneric,
                    "tapir reduction loop must be a top-level loop");

  return true;
}
