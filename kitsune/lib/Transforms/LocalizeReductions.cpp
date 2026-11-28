//===- LocalizeReductions.cpp - Compute local reductions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compute per-thread reductions in prepared tapir loops. At the end of the
// thread, accumulate this contribution to the original destination. Consider
// the tapir reduction loop shown below:
//
//     int res = <identity>;
//     forall(...) {
//       __kitsune_reduce(&res, ...)
//     }
//
// This will be transformed to the following:
//
//     int res = <identity>;
//     forall(...) {
//       int local = <identity>;
//       __kitsune_reduce(&local, ...)
//       __kitsune_reduce(&res, local, ...);
//     }
//
// From the example above, it is not clear why this transformation is
// beneificial. However, in general, the reduction could be performed inside
// nested serial loops within the parallel loop body. In such cases, performing
// a local reduction, which can be performed in serial, followed by a parallel
// reduction into the actual result can be more efficient.
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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/IRBuilder.h"

#define DEBUG_TYPE "kit-reduce-localize"

using namespace llvm;

namespace {

class LocalizeReductions {
public:
  // Map from the destinations of reductions to the local into which they will
  // be reduced before performing the final reduction.
  using Locals = DenseMap<Value *, AllocaInst *>;

protected:
  LoopInfo &li;

protected:
  Locals genLocals(Loop &loop, const Reductions &redxns);
  void genReduceIntoLocal(Loop &loop, const ReductionInfo &redxn,
                          const Locals &locals);
  void genReduceIntoFinal(Loop &loop, const ReductionInfo &redxn,
                          const Locals &locals);
  void updateCallAttributes(Loop &loop, const ReductionInfo &redxn);
  bool localize(Loop &loop, const ReductionInfo &redxn, const Locals &locals);
  bool localizeCPU(Loop &loop, const ReductionInfo &redxn,
                   const Locals &locals);
  bool localizeGPU(Loop &loop, const ReductionInfo &redxn,
                   const Locals &locals);
  bool check(const Loop &loop, const Reductions &redxns);
  bool run(Loop &loop, const Reductions &redxns);
  bool run(Loop &loop);

public:
  LocalizeReductions(LoopInfo &li) : li(li) {}

  bool run(Function &f);
};

} // namespace

LocalizeReductions::Locals
LocalizeReductions::genLocals(Loop &loop, const Reductions &redxns) {
  // Get the point at which to insert allocas for the local reductions. We
  // cannot use BasicBlock::getFirst* functions because the variant that looks
  // for allocas only works if the basic block is the function entry block.
  auto getInsertPoint = [](BasicBlock &bb) -> Instruction * {
    for (Instruction &inst : bb)
      if (!isa<AllocaInst>(inst) && !isa<PHINode>(inst))
        return &inst;
    llvm_unreachable("No valid insert point in basic block");
  };

  LLVMContext &ctx = getContext(loop);
  IRBuilder<> builder(ctx);

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
  BasicBlock *bbBody = detach->getDetached();

  Locals locals;
  for (const ReductionInfo &redxn : redxns) {
    Value *dest = redxn.getDest();
    Value *unit = redxn.getUnit();
    if (!locals.contains(dest)) {
      builder.SetInsertPoint(getInsertPoint(*bbBody));
      AllocaInst *local = builder.CreateAlloca(redxn.getResultBufferType());
      if (isa<PointerType>(redxn.getType())) {
        MaybeAlign align;
        builder.CreateMemCpy(local, align, unit, align, redxn.elemSize);
      } else {
        builder.CreateStore(unit, local);
      }
      locals[dest] = local;
    }
  }
  return locals;
}

void LocalizeReductions::genReduceIntoLocal(Loop &loop,
                                            const ReductionInfo &redxn,
                                            const Locals &locals) {
  // Since we are guaranteed that the destination of the reduction is not used
  // elsewhere in the loop (except perhaps as the destination of other
  // reductions in the loop), to reduce into the local instead, we only need to
  // replace the destination in the call to the reduce intrinsic. We also update
  // the reduction strategy to naive since this is always a straightforward
  // serial reduction.
  LLVMContext &ctx = getContext(loop);
  Value *strategy = toConstant(ReductionStrategy::Naive, ctx);
  AllocaInst *local = locals.at(redxn.getDest());

  redxn.call->setArgOperand(ReductionInfo::argNumDest, local);
  redxn.call->setArgOperand(ReductionInfo::argNumStrategy, strategy);
}

void LocalizeReductions::genReduceIntoFinal(Loop &loop,
                                            const ReductionInfo &redxn,
                                            const Locals &locals) {
  // The reattach instruction is guaranteed to be the very last instruction in
  // the loop body. Once outlined, any instructions added right before the
  // reattach instruction will effectively be the very last instructions in the
  // outlined function, except for the return. The accumulation into the final
  // destination can be safely performed there where it is guaranteed to be
  // executed after the local reduction has been computed.
  ReattachInst *reattach = getUniqueInstInLoopOnly<ReattachInst>(loop);
  IRBuilder<> builder(reattach);

  // Now add the final reduction call.
  Value *tt = redxn.getTTV();
  Value *op = redxn.getReduceOpV();
  Value *dest = redxn.getDest();
  Value *size = redxn.getElemSizeV();
  Value *unit = redxn.getUnit();
  Value *reducer = redxn.getReducer();
  Value *strategy = redxn.getStrategyV();
  Type *type = redxn.getType();
  AllocaInst *local = locals.at(dest);

  LoadInst *value = builder.CreateLoad(type, local);
  SmallVector<Type *, 2> overloadTys = redxn.getOverloadTypes();
  SmallVector<Value *, 8> args = {tt,       op,   dest, value,
                                  strategy, size, unit, reducer};
  args.append(redxn.getExtraArgs());
  builder.CreateIntrinsic(Intrinsic::kit_reduce, overloadTys, args);
}

void LocalizeReductions::updateCallAttributes(Loop &loop,
                                              const ReductionInfo &redxn) {
  // The original reduction call may have been marked as tail since the
  // destination of the reduction may be not be a stack pointer. However,
  // after reducing into an alloca, it is no longer safe for this call to be
  // a tail call.
  redxn.call->setTailCall(CallInst::TCK_None);
}

bool LocalizeReductions::localize(Loop &loop, const ReductionInfo &redxn,
                                  const Locals &locals) {
  // Generate the final reduction. We do this first because we need the original
  // destination, which will eventually be replaced with the thread-local stack
  // slot that was allocated for it.
  genReduceIntoFinal(loop, redxn, locals);

  // It is now safe to update the original reduction to reduce into the local
  // variable that was allocated. This will also update the reduction strategy.
  // After this, it is no longer safe to use the ReductionInfo object.
  genReduceIntoLocal(loop, redxn, locals);

  // Update original reduction attributes.
  updateCallAttributes(loop, redxn);

  return true;
}

bool LocalizeReductions::localizeCPU(Loop &loop, const ReductionInfo &redxn,
                                     const Locals &locals) {
  return localize(loop, redxn, locals);
}

bool LocalizeReductions::localizeGPU(Loop &root, const ReductionInfo &redxn,
                                     const Locals &locals) {
  Loop *loop = getInnermostTapirLoopInNest(root);
  return localize(*loop, redxn, locals);
}

bool LocalizeReductions::check(const Loop &loop, const Reductions &redxns) {
  if (!loop.isLoopSimplifyForm())
    return emitDiagnostic(loop, DiagID::ErrLoopNotSimplifyForm);

  for (const ReductionInfo &redxn : redxns)
    if (std::optional<DiagID> err = redxn.checkUsesOfDest(li))
      return emitDiagnostic(loop, *err, getName(*redxn.getDest()));

  return true;
}

bool LocalizeReductions::run(Loop &loop) {
  assert(isTapirLoop(loop) && "Loop is a tapir loop");
  assert(hasReductionAttr(loop) && "Loop is a reduction loop");
  assert(getUniqueInstInLoopOnly<DetachInst>(loop) &&
         "Tapir loop has a unique detach instruction");
  assert(getUniqueInstInLoopOnly<ReattachInst>(loop) &&
         "Tapir loop has a unique reattach instruction");

  TTID tt = *getTargetAttr(loop);

  const Reductions redxns = getReductions(loop);
  if (!check(loop, redxns))
    return false;

  // If the destination of a reduction is not loop-invariant, it cannot be
  // localized since we cannot know how much space to allocate for the local
  // reductions. We also don't localize when the loop has the serial tapir
  // target since it is not clear if there is any benefit in doing so.
  Reductions localizable;
  for (const ReductionInfo &redxn : redxns) {
    Value *dest = redxn.getDest();
    if (tt != TTID::Serial && tt != TTID::Custom && loop.isLoopInvariant(dest))
      localizable.push_back(redxn);
  }

  // Generate allocas for the local reduction. We generate one alloca per
  // destination - not one alloca per reduction.
  Locals locals = genLocals(*getInnermostTapirLoopInNest(loop), localizable);

  bool changed = false;
  for (const ReductionInfo &redxn : localizable) {
    if (loop.isLoopInvariant(redxn.getDest())) {
      if (tt == TTID::Serial || tt == TTID::Custom)
        changed |= false;
      else if (isCPUTT(tt))
        changed |= localizeCPU(loop, redxn, locals);
      else if (isGPUTT(tt))
        changed |= localizeGPU(loop, redxn, locals);
      else
        llvm_unreachable("Unknown tapir target category");
    }
  }

  // All loops are marked as localized, even if nothing was changed. This pass
  // may be run more than once, so subsequent invocations should ignore loops
  // that have already been processed.
  for (Loop *l : getTapirLoopsInNest(loop))
    addLocalizedAttr(*l);

  return changed;
}

bool LocalizeReductions::run(Function &f) {
  bool changed = false;
  for (Loop *loop : getTopLevelTapirLoops(li))
    if (hasReductionAttr(*loop) && !hasLocalizedAttr(*loop))
      changed |= run(*loop);
  return changed;
}

PreservedAnalyses LocalizeReductionsPass::run(Function &f,
                                              FunctionAnalysisManager &am) {
  LoopInfo &li = am.getResult<LoopAnalysis>(f);

  if (LocalizeReductions(li).run(f))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
