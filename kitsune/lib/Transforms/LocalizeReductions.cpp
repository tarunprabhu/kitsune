//===- LocalizeReductions.cpp - Compute local reductions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compute per-thread reductions in prepared tapir loops. At the end of the
// thread, accumulate this contribution to the original destination.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/LocalizeReductions.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/Reductions.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Core/ValueUtils.h"
#include "kitsune/Shared/RTNestUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/IRBuilder.h"

#define DEBUG_TYPE "kit-reduce-localize"

using namespace llvm;

namespace {

class LocalizeReductionsImpl {
public:
  using Reductions = SmallVector<ReductionInfo, 1>;

protected:
  LoopInfo &li;

protected:
  AllocaInst *genLocal(const ReductionInfo &redxn, Instruction *insertPt);
  void genReduceIntoLocal(const ReductionInfo &redxn, AllocaInst *local);
  void genReduceIntoFinal(const ReductionInfo &redxn, AllocaInst *local,
                          Instruction *insertPt);
  bool localizeReduction(Loop &loop, const Reductions &redxns);
  bool localizeReductionCPU(Loop &loop, const Reductions &redxns);
  bool localizeReductionGPU(Loop &loop, const Reductions &redxns);
  bool localizeReductionSerial(Loop &loop, const Reductions &redxns);
  bool check(const Loop &loop, const Reductions &redxns);
  bool run(Loop &loop);

public:
  LocalizeReductionsImpl(LoopInfo &li) : li(li) {}

  bool run(Function &f);
};

} // namespace

AllocaInst *LocalizeReductionsImpl::genLocal(const ReductionInfo &redxn,
                                             Instruction *insertPt) {
  Type *type = redxn.getType();
  Value *unit = redxn.getUnit();
  unsigned size = redxn.elemSize;
  Type *localType = redxn.getResultBufferType();

  IRBuilder<> builder(insertPt);
  AllocaInst *local = builder.CreateAlloca(localType, /*ArraySize=*/nullptr);
  if (isa<PointerType>(type))
    builder.CreateMemCpy(local, MaybeAlign(), unit, MaybeAlign(), size);
  else
    builder.CreateStore(unit, local);

  return local;
}

void LocalizeReductionsImpl::genReduceIntoLocal(const ReductionInfo &redxn,
                                                AllocaInst *local) {
  // Since we are guaranteed that the destination of the reduction is not used
  // elsewhere in the loop (except perhaps as the destination of other
  // reductions in the loop), to reduce into the local instead, we only need to
  // replace the destination in the call to the reduce intrinsic. We also update
  // the reduction strategy to naive since this is always a straightforward
  // serial reduction.
  LLVMContext &ctx = local->getContext();
  Value *strategy = toConstant(ReductionStrategy::Naive, ctx);

  redxn.call->setArgOperand(ReductionInfo::argNumDest, local);
  redxn.call->setArgOperand(ReductionInfo::argNumStrategy, strategy);
}

void LocalizeReductionsImpl::genReduceIntoFinal(const ReductionInfo &redxn,
                                                AllocaInst *local,
                                                Instruction *insertPt) {
  Value *tt = redxn.getTTV();
  Value *op = redxn.getReduceOpV();
  Value *dest = redxn.getDest();
  Type *type = redxn.getType();
  Value *size = redxn.getElemSizeV();
  Value *unit = redxn.getUnit();
  Value *reducer = redxn.getReducer();
  Value *strategy = redxn.getStrategy();

  IRBuilder<> builder(insertPt);
  Value *value = builder.CreateLoad(type, local);

  SmallVector<Type *, 2> overloadTys = redxn.getOverloadTypes();
  SmallVector<Value *, 8> args = {tt,       op,   dest, value,
                                  strategy, size, unit, reducer};
  args.append(redxn.getExtraArgs());
  builder.CreateIntrinsic(Intrinsic::kit_reduce, overloadTys, args);
}

bool LocalizeReductionsImpl::localizeReduction(Loop &loop,
                                               const Reductions &redxns) {
  // Get the point at which to insert allocas for the local reductions. We
  // cannot use BasicBlock::getFirst* functions because the variant that looks
  // for allocas only works if the basic block is the function entry block.
  auto getInsertPoint = [](BasicBlock &bb) -> Instruction * {
    for (Instruction &inst : bb)
      if (!isa<AllocaInst>(inst) && !isa<PHINode>(inst))
        return &inst;
    llvm_unreachable("No valid insert point in basic block");
  };

  DenseMap<Value *, AllocaInst *> locals;

  // Generate allocas for the local reduction. We generate one alloca per
  // destination - not one alloca per reduction.
  //
  // We add the alloca to the detached block of the tapir loop. This is
  // essentially the "entry block" of the loop body. These tapir loops will
  // eventually be outlined. Tapir loops with a CPU-centric tapir target are
  // guaranteed to be of depth 1, with exactly one iteration of the tapir loop
  // executing per thread. As a result, any allocas that are in the body of the
  // loop will end up in the function entry block after outlining. On GPU's, the
  // loop whose reductions are localized will be the innermost loop in the nest.
  // There, too, exactly one iteration of the nest is executed per GPU thread,
  // so the alloca will again end up in the entry block of the kernel function.
  // We don't create a local alloca for each reduction, but for each unique
  // value that is used as the destination of a reduction. This ensures that we
  // have exactly as many local allocas as there are unique destinations of
  // reductions.
  DetachInst *detach = getUniqueInstInLoopOnly<DetachInst>(loop);
  BasicBlock &bbBody = *detach->getDetached();
  for (const ReductionInfo &redxn : redxns) {
    Value *dest = redxn.getDest();
    if (!locals.contains(dest))
      locals[dest] = genLocal(redxn, getInsertPoint(bbBody));
  }

  // Generate the final reduction. We do this first because we need the original
  // destination, which will eventually be replaced with the thread-local stack
  // slot that was allocated for it.
  //
  // The reattach instruction is guaranteed to be the very last instruction in
  // the loop body. Once outlined, any instructions added right before the
  // reattach instruction will be the very last instructions in the outlined
  // function, except for the return. The accumulation into the final
  // destination can be safely performed there where it is guaranteed to be
  // executed after the local reduction has been computed.
  ReattachInst *reattach = getUniqueInstInLoopOnly<ReattachInst>(loop);
  for (const ReductionInfo &redxn : redxns)
    genReduceIntoFinal(redxn, locals.at(redxn.getDest()), reattach);

  // It is now safe to update the original reduction to reduce into the local
  // variable that was allocated. This will also update the reduction strategy.
  // After this, it is no longer safe to use the ReductionInfo object.
  for (const ReductionInfo &redxn : redxns)
    genReduceIntoLocal(redxn, locals.at(redxn.getDest()));

  return true;
}

bool LocalizeReductionsImpl::localizeReductionCPU(Loop &loop,
                                                  const Reductions &redxns) {
  bool changed = localizeReduction(loop, redxns);
  addLocalizedAttr(loop);
  return changed;
}

bool LocalizeReductionsImpl::localizeReductionGPU(Loop &root,
                                                  const Reductions &redxns) {
  SmallVector<Loop *, 0> loops = getTapirLoopsInNest(root);
  bool changed = localizeReduction(*loops.back(), redxns);
  for (Loop *loop : loops)
    addLocalizedAttr(*loop);
  return changed;
}

bool LocalizeReductionsImpl::localizeReductionSerial(Loop &loop,
                                                     const Reductions &redxns) {
  // For tapir reduction loops with the serial tapir target, we don't do
  // anything.
  return false;
}

bool LocalizeReductionsImpl::check(const Loop &loop, const Reductions &redxns) {
  if (!loop.isLoopSimplifyForm())
    return emitDiagnostic(loop, DiagID::ErrLoopNotSimplifyForm);

  for (const ReductionInfo &redxn : redxns)
    if (std::optional<DiagID> err = redxn.checkUsesOfDest(li))
      return emitDiagnostic(loop, *err, getName(*redxn.getDest()));

  return true;
}

bool LocalizeReductionsImpl::run(Loop &loop) {
  assert(isTapirLoop(loop) && "Loop is a tapir loop");
  assert(hasReductionAttr(loop) && "Loop is a reduction loop");
  assert(getUniqueInstInLoopOnly<DetachInst>(loop) &&
         "Tapir loop has a unique detach instruction");
  assert(getUniqueInstInLoopOnly<ReattachInst>(loop) &&
         "Tapir loop has a unique reattach instruction");

  Reductions redxns = collectReductions(loop);
  if (!check(loop, redxns))
    return false;

  TTID tt = *getTargetAttr(loop);
  if (tt == TTID::Serial)
    return localizeReductionSerial(loop, redxns);
  else if (isCPUTT(tt))
    return localizeReductionCPU(loop, redxns);
  else if (isGPUTT(tt))
    return localizeReductionGPU(loop, redxns);
  else if (tt == TTID::Custom)
    return false;
  else
    llvm_unreachable("Unknown tapir target category");
}

bool LocalizeReductionsImpl::run(Function &f) {
  bool changed = false;
  for (Loop *loop : getTopLevelTapirLoops(li))
    if (hasReductionAttr(*loop) && !hasLocalizedAttr(*loop))
      changed |= run(*loop);
  return changed;
}

PreservedAnalyses LocalizeReductionsPass::run(Function &f,
                                              FunctionAnalysisManager &am) {
  LoopInfo &li = am.getResult<LoopAnalysis>(f);

  if (LocalizeReductionsImpl(li).run(f))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
