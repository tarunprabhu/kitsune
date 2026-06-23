//===- PreLowerVerification.cpp - Verification before tapir lowering ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass that checks the tapir targets that are required by a module, and
// the structure of tapir tasks in the functions. This is primarily a sanity
// check just before we begin lowering tapir loops. It is, therefore, intended
// to be run relatively late in the overall pipeline, but fairly early in the
// lowering pipeline.
//
// One of the main analyses that this performs on a function is checking the
// tapir targets associated with tapir loops. For instance, an error will be
// raised in the following case:
//
//     parallel_for (...) {      // tapir.loop.target = "cuda"
//        parallel_for (...) {   // tapir.loop.target = "hip"
//        }
//     }
//
// The issue here is not that multiple targets are being used, but that any
// lowering for this would require an NVIDIA GPU to launch a kernel on an
// AMDGPU. It is highly unlikely that this can ever be made to work.
//
// This pass emits diagnostics (warnings and errors) to stderr. The default
// behavior is to exit with a system-dependent error code if at least one error
// was found. This can be overridden to continue as normal even if errors were
// found. At some point, this may change to an analysis pass that returns the
// results of the analysis.
//
//
// NOTES FOR MAINTAINERS
//
//  1. The tapir-target-analysis pass looks over the tapir loops and collects
//     the tapir targets that are needed, but does not perform any checks. There
//     is an open question on whether the tapir target checks should be moved
//     there leaving only the structure checks here. Alternatively, that pass
//     could be seen as a simple wrapper for the instances of the tapir target
//     objects that will be needed during lowering.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Analysis/PreLowerVerification.h"
#include "kitsune/Analysis/TTObjectsAnalysis.h"
#include "kitsune/Analysis/TapirLoopNestAnalysis.h"
#include "kitsune/Core/BasicBlockUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "kitsune/Support/ErrorHandling.h"
#include "kitsune/Targets/TapirTargets.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "kit-verify-prelower"

using namespace llvm;

static cl::opt<bool> clDisableVerifyPreLower(
    "kit-no-verify-prelower", cl::init(false), cl::Hidden,
    cl::cat(cl::catKitClDevOpts),
    cl::desc("Disable Kitsune's pre-lowering verifier"));

namespace {

static SmallSet<const SyncInst *, 4> getSyncInstsFor(const Value &syncRegion) {
  SmallSet<const SyncInst *, 4> syncInsts;
  for (const Use &use : syncRegion.uses())
    if (const auto *syncInst = dyn_cast<SyncInst>(use.getUser()))
      syncInsts.insert(syncInst);
  return syncInsts;
}

// Result of running the tapir verifier. Currently, this only contains counts
// of the number of errors and warnings that were emitted.
struct Log {
  unsigned errors = 0;
  unsigned warnings = 0;
};

template <typename T> class Verifier {
protected:
  Log &log;

protected:
  Verifier(Log &log) : log(log) {}

  void record(DiagID id) {
    if (isError(id))
      ++log.errors;
    else if (isWarning(id))
      ++log.warnings;
  }

  template <typename... Args> bool emitDiag(DiagID id, Args &&...args) {
    emitDiagnostic(id, args...);
    record(id);
    return false;
  }

  template <typename IRElement, typename... Args>
  bool emitDiag(const IRElement &e, DiagID id, Args &&...args) {
    emitDiagnostic(e, id, args...);
    record(id);
    return false;
  }
};

// Verifier class for a function. This is only a class because it is a
// convenient container for the function-level analyses that the various
// checks may use.
class VerifierF : public Verifier<VerifierF> {
private:
  LoopInfo &li;
  OptimizationRemarkEmitter &ore;
  PostDominatorTree &pdt;
  ScalarEvolution &se;
  TaskInfo &ti;

private:
  // Check that the tapir targets on all subloops in a tapir loop nest rooted at
  // the given loop are consistent. The target of the root must be a GPU-centric
  // tapir target.
  void checkConsistentTTsForGPU(Loop &root) {
    TTID ttRoot = *getTargetAttr(root);
    for (Loop *subLoop : getAllSubLoops(root)) {
      if (isTapirLoop(*subLoop)) {
        TTID tt = *getTargetAttr(*subLoop);
        if (tt != ttRoot) {
          emitDiag(*subLoop, DiagID::ErrTTIncompatibleLoopGPU, tt, ttRoot);
          emitDiag(root, DiagID::NoteAncestorLoopTarget, ttRoot);
        }
      }
    }
  }

  // Check that the tapir targets on all subloops in a tapir loop nest rooted at
  // the given loop are consistent. The target of the root must be a CPU-centric
  // tapir target.
  void checkConsistentTTsForCPU(Loop &root) {
    TTID ttRoot = *getTargetAttr(root);
    for (Loop *subLoop : getAllSubLoops(root)) {
      if (isTapirLoop(*subLoop)) {
        TTID tt = *getTargetAttr(*subLoop);
        if (isGPUTT(tt) || tt != ttRoot) {
          // FIXME: We don't yet support multi-target compilation anyway, but
          // GPU's inside parallel CPU loops are particularly thorny. Until we
          // have a decent plan for handling these, complain about them.
          //
          // Although nesting CPU targets should be ok, we need to think about
          // the consequences of these, so don't allow those either.
          emitDiag(*subLoop, DiagID::ErrTTIncompatibleLoop, tt, ttRoot);
          emitDiag(root, DiagID::NoteAncestorLoopTarget, ttRoot);
        }
      }
    }
  }

  // If the root of a tapir loop nest is a GPU-centric tapir target, any tapir
  // loops contained within it must be perfectly nested. Otherwise, they are
  // likely to be serialized.
  void checkLoopNestStructureForGPU(Loop &root) {
    std::unique_ptr<TapirLoopNest> nest = TapirLoopNest::create(root, se);
    assert(nest && "Could not create tapir loop nest object");

    ArrayRef<Loop *> perfectLoops = nest->getPerfectTapirLoops();
    SmallSetVector<Loop *, 4> perfectSet(perfectLoops.begin(),
                                         perfectLoops.end());
    for (Loop *loop : nest->getLoops()) {
      if (isTapirLoop(*loop)) {
        if (!perfectSet.contains(loop)) {
          emitDiag(*loop, DiagID::WarnParallelLoopImperfectlyNested);
          emitDiag(DiagID::NoteLoopNestRoot, getName(root));
        }
      }
    }

    // The loop bounds of all perfectly nested tapir loops in a tapir loop nest
    // must be loop-invariant with respect to the outer loop.
    for (const Loop *loop : perfectLoops) {
      std::optional<Loop::LoopBounds> maybeLB = loop->getBounds(se);
      assert(maybeLB && "Could not get bounds for loop");

      Loop::LoopBounds lb = *maybeLB;
      if (!root.isLoopInvariant(&lb.getFinalIVValue())) {
        emitDiag(*loop, DiagID::ErrTapirNestBoundsVariantGPU);
        emitDiag(DiagID::NoteLoopNestRoot, getName(root));
      }
    }
  }

  void checkTopLevelTapirLoop(Loop &loop) {
    switch (*getTargetAttr(loop)) {
    case TTID::Nolo:
    case TTID::Serial:
    case TTID::OpenCilk:
    case TTID::OpenMP:
    case TTID::Pthreads:
    case TTID::Qthreads:
      checkConsistentTTsForCPU(loop);
      return;
    case TTID::Cuda:
    case TTID::Hip:
      checkConsistentTTsForGPU(loop);
      checkLoopNestStructureForGPU(loop);
      return;
    case TTID::Custom:
      // FIXME: We should probably require the custom targets to have a hook
      // that provides suitable checks.
      return;
    case TTID::Lambda:
    case TTID::OMPTask:
    case TTID::Realm:
      break;
    }
    llvm_unreachable("checkTopLevelTapirLoop: TTID not handled");
  }

  // Find the top-level tapir loops in a function and check that they are
  // consistent. This primarily checks the tapir targets on the subloops and
  // the loop nest structure.
  void checkTopLevelTapirLoops(Function &f) {
    for (Loop *loop : getTopLevelTapirLoops(li))
      checkTopLevelTapirLoop(*loop);
  }

  // Check the instructions in the tapir loop preheader.
  //
  //   - It must be terminated by an unconditional branch instruction.
  //
  bool checkTapirLoopPreheader(Loop &loop) {
    BasicBlock *ph = loop.getLoopPreheader();
    if (!isUncondBr(*ph->getTerminator()))
      return emitDiag(loop, DiagID::ErrTapirLoopBlockTerminator, "preheader",
                      "unconditional branch");
    return true;
  }

  // Check the instructions in the tapir loop header.
  //
  //  - It must be terminated by a detach instruction.
  //
  //  - It can only contain PHI nodes, and no other instructions.
  //
  bool checkTapirLoopHeader(Loop &loop) {
    BasicBlock *header = loop.getHeader();

    // Debug info instructions are permitted. However, we expect these to be
    // automatically upgraded to DbgRecord's, so we don't check for them.
    for (Instruction &inst : *header)
      if (!isa<PHINode>(inst) && !isa<DetachInst>(inst))
        return emitDiag(loop, DiagID::ErrTapirLoopHeaderInstNotPHI);

    if (!isa<DetachInst>(header->getTerminator()))
      return emitDiag(loop, DiagID::ErrTapirLoopBlockTerminator, "header",
                      "detach");

    return true;
  }

  // Check the instructions in the tapir loop latch.
  //
  //  - It must be terminated by a conditional branch instruction.
  //
  //  - It can only contain the instructions to update the canonical loop
  //    induction variable, and nothing else.
  //
  bool checkTapirLoopLatch(Loop &loop) {
    BasicBlock *latch = loop.getLoopLatch();
    Instruction *latchTerm = latch->getTerminator();
    if (!isCondBr(*latchTerm))
      return emitDiag(loop, DiagID::ErrTapirLoopBlockTerminator, "latch",
                      "conditional branch");

    // Collect the instructions that are used to compute the termination
    // condition and branch.
    SmallPtrSet<Instruction *, 4> expected;
    BranchInst *br = cast<BranchInst>(latchTerm);
    if (auto *cmp = dyn_cast<Instruction>(br->getCondition())) {
      for (Value *op : cmp->operands())
        if (auto *inst = dyn_cast<Instruction>(op))
          expected.insert(inst);
      expected.insert(cmp);
    }
    expected.insert(br);

    // In some cases, the updated induction variable is not actually used to
    // compute the termination condition. This can happen, for instance, when
    // the loop induction variable is in the range [1,n], in which case the
    // termination condition in the loop is computed using the "old" value of
    // the canonical loop induction, i, not the "updated" value of i. In other
    // words, while one might expect to see the following code in the latch:
    //
    //   %next.i = add i64 %i, 1
    //   %cmp.i = icmp eq i64 %next.i, %n
    //   br label %cmp.i, label %exit, label %header
    //
    // we instead see something like this:
    //
    //   %next.i = add i64 %i, 1
    //   %cmp.i = icmp eq i64 %i, %n1
    //   br label %cmp.i, label %exit, label %header
    //
    // where %n1 = sub i64 %n, 1.
    //
    // To handle this case, we unconditionally add the incoming value from the
    // latch to the expected instructions list. It is safe to cast this to an
    // instruction because we are guaranteed that it is incremented by 1
    //
    PHINode *primIV = loop.getCanonicalInductionVariable();
    expected.insert(cast<Instruction>(primIV->getIncomingValueForBlock(latch)));

    // Debug info instructions are permitted. However, we expect these to be
    // automatically upgraded to DbgRecord's, so we don't check for them.
    for (Instruction &inst : *latch)
      if (!expected.contains(&inst))
        return emitDiag(loop, DiagID::ErrTapirLoopLatchInstUnexpected,
                        getName(inst));

    return true;
  }

  // Check that a given value is a sync region definition.
  template <typename InstType> bool checkSyncRegionDefn(const InstType &inst) {
    if (const auto *call = dyn_cast<CallBase>(inst.getSyncRegion()))
      if (Function *f = call->getCalledFunction())
        if (f->getIntrinsicID() == Intrinsic::syncregion_start)
          return true;
    return emitDiag(inst, DiagID::ErrTapirLoopSyncRegionDefn);
  }

  // Check the instructions in a tapir loop.
  //
  //  - The loop must contain exactly one detach instruction, and exactly one
  //    reattach instruction (subloops may also contain detaches and
  //    reattaches, but these are ignored). Specifically, we don't allow
  //    "free-standing" detaches and reattaches, although Tapir allows them.
  //    This is because this represents a form of nested parallelism that we
  //    don't yet support.
  //
  //  - A sync instruction must post-dominate the tapir loop.
  //
  //  - The terminator of the loop preheader must be an unconditional branch.
  //
  //  - The terminator of the loop latch must be a conditional branch. This
  //    should nearly always be the case, but changes to the pass pipeline may
  //    result in this constraint being violated.
  //
  bool checkTapirLoopInsts(const Loop &loop, Task &task) {
    const DetachInst *detachInst = task.getDetach();
    if (getUniqueInstInLoopOnly<DetachInst>(loop) != detachInst)
      return emitDiag(loop, DiagID::ErrTapirLoopNoUniqueDetachInst);

    const auto *reattachInst = getUniqueInstInLoopOnly<ReattachInst>(loop);
    if (!reattachInst)
      return emitDiag(loop, DiagID::ErrTapirLoopNoUniqueReattachInst);

    const Value *syncRegion = detachInst->getSyncRegion();
    SmallSet<const SyncInst *, 4> syncInsts = getSyncInstsFor(*syncRegion);
    if (syncInsts.size() < 1) {
      return emitDiag(loop, DiagID::ErrTapirLoopNoUniqueSyncInst);
    } else if (syncInsts.size() > 1) {
      // We could have multiple sync instructions in a sync since task-simplify
      // may have merged sync regions. Ideally, we would not want this to be an
      // error, but we will have to fix the optimization pipeline before we can
      // do that.
      return true;
    }

    // We have to check that the sync instruction post-dominates the loop. We
    // do this by checking that each of the loop exit blocks is post-dominated
    // by the sync instruction. We expect that the tapir loop is in
    // loop-simplify form at this time. This would imply that all exit blocks
    // are dominated by the loop header. Therefore, if the body of the loop is
    // entered, and a sync instruction post-dominates all exits, then the sync
    // is guaranteed to be encountered when the loop body is exited.
    //
    // When doing so, we deliberately skip "unreachable" blocks i.e. those
    // blocks that consist of a single unreachable instruction. We have to do
    // this as a special case because the presence of such blocks will cause
    // the post-dominator tree to report that the sync does not post-dominate
    // the exit. While the semantics of LLVM's unreachable instruction are not
    // formally specified at the time of writing this, it is reasonable to
    // assume that encountering it at runtime will result in a catastrophic
    // failure. In the CFG, therefore, there can be no path from there to a sync
    // instruction.
    const SyncInst *syncInst = *syncInsts.begin();
    for (BasicBlock *exit : getUniqueExitBlocks(loop))
      if (!isDeadEnd(*exit) && !pdt.dominates(syncInst, &exit->front()))
        return emitDiag(loop, DiagID::ErrTapirLoopSyncMustPostDominate);

    checkSyncRegionDefn(*detachInst);
    checkSyncRegionDefn(*reattachInst);

    return true;
  }

  // Check that the tapir loop has a single induction variable, and that that IV
  // is canonical.
  bool checkTapirLoopIV(Loop &loop) {
    if (getNumIndVars(loop) > 1)
      return emitDiag(loop, DiagID::ErrTapirLoopIVMultiple);

    if (!loop.getCanonicalInductionVariable())
      return emitDiag(loop, DiagID::ErrTapirLoopIVNotCanonical);

    return true;
  }

  // Check additional properties of tapir loops that are derived from it.
  //
  //  - The tapir loop has a finite trip count
  //
  bool checkTapirLoopProperties(Loop &loop, Task &task) {
    PredicatedScalarEvolution pse(se, loop);
    TapirLoopInfo tl(&loop, &task);
    tl.collectIVs(pse, DEBUG_TYPE, &ore);
    if (!tl.getOrCreateTripCount(pse, DEBUG_TYPE, &ore))
      return emitDiag(loop, DiagID::ErrTapirLoopNoFiniteTripCount);

    return true;
  }

  bool checkAllTapirLoops(Function &f) {
    for (Loop *loop : li.getLoopsInPreorder()) {
      if (!isTapirLoop(*loop))
        continue;

      Task *task = getTaskIfTapirLoop(loop, &ti);
      if (!task)
        return emitDiag(*loop, DiagID::ErrTapirLoopNoTask);

      if (!loop->isLoopSimplifyForm())
        return emitDiag(*loop, DiagID::ErrLoopNotSimplifyForm);

      // clang-format off
      if (!checkTapirLoopIV(*loop)
          || !checkTapirLoopInsts(*loop, *task)
          || !checkTapirLoopPreheader(*loop)
          || !checkTapirLoopHeader(*loop)
          || !checkTapirLoopLatch(*loop)
          || !checkTapirLoopProperties(*loop, *task))
        return false;
      // clang-format on
    }

    return true;
  }

  // Check that detach and reattach instructions do not appear outside tapir
  // loops.
  void checkTapirInsts(const Function &f) {
    SmallSet<const BasicBlock *, 8> bbs;
    for (const Loop *loop : getTapirLoops(li))
      for (const BasicBlock *bb : getBlocksNotInSubLoops(*loop))
        bbs.insert(bb);

    for (const_inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i) {
      if (isa<DetachInst>(*i) || isa<ReattachInst>(*i)) {
        if (!bbs.contains(i->getParent()))
          emitDiag(*i, DiagID::ErrTapirInstNotInTapirLoop);
      } else if (const auto *syncInst = dyn_cast<SyncInst>(&*i)) {
        checkSyncRegionDefn(*syncInst);
      }
    }
  }

public:
  VerifierF(Log &log, LoopInfo &li, OptimizationRemarkEmitter &ore,
            PostDominatorTree &pdt, ScalarEvolution &se, TaskInfo &ti)
      : Verifier(log), li(li), ore(ore), pdt(pdt), se(se), ti(ti) {}

  void run(Function &f) {
    checkTapirInsts(f);
    checkAllTapirLoops(f);
    checkTopLevelTapirLoops(f);
  }
};

// Verifier for a module. This does not descend into any of the functions, but
// only verifies module-level entities such as global variables, module-level
// debug information and metadata, etc.
class VerifierM : public Verifier<VerifierM> {
private:
  TTObjects &ttObjs;

private:
  // Check if all tapir targets required by the module have been enabled.
  void checkTTsEnabled(ArrayRef<TTID> tts) {
    for (TTID tt : tts)
      if (not isTTEnabled(tt))
        emitDiag(DiagID::ErrTTNotEnabled);
  }

public:
  VerifierM(Log &log, TTObjects &ttObjs) : Verifier(log), ttObjs(ttObjs) {}

  void run(Module &m) {
    ArrayRef tts = ttObjs.getRequiredTTs(m);
    checkTTsEnabled(tts);

    // At this time, we do not support multi-target execution. Therefore, only
    // one tapir target must be required in the module. Once we support
    // multi-target execution, this check can be removed.
    if (tts.size() > 1)
      emitDiag(DiagID::ErrTTMultiple);
  }
};

} // namespace

PreservedAnalyses PreLowerVerificationPass::run(Module &m,
                                                ModuleAnalysisManager &mam) {
  TTObjects &ttObjs = mam.getResult<TTObjectsAnalysis>(m);
  if (clDisableVerifyPreLower || !ttObjs.hasTTID())
    return PreservedAnalyses::all();

  FunctionAnalysisManager &fam =
      mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();

  Log log;
  VerifierM(log, ttObjs).run(m);
  for (Function &f : m) {
    if (f.size()) {
      LoopInfo &li = fam.getResult<LoopAnalysis>(f);
      OptimizationRemarkEmitter &ore =
          fam.getResult<OptimizationRemarkEmitterAnalysis>(f);
      PostDominatorTree &pdt = fam.getResult<PostDominatorTreeAnalysis>(f);
      ScalarEvolution &se = fam.getResult<ScalarEvolutionAnalysis>(f);
      TaskInfo &ti = fam.getResult<TaskAnalysis>(f);

      VerifierF(log, li, ore, pdt, se, ti).run(f);
    }
  }

  if (log.errors and exitIfError)
    exitOnError();
  return PreservedAnalyses::all();
}
