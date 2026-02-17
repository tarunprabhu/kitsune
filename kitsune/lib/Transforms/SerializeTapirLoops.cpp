//===- SerializeTapirLoops.cpp - Serialize certain tapir loops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to analyze tapir loop nests and serialize certain tapir loops.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/SerializeTapirLoops.h"
#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Utils/TaskSimplify.h"

#define DEBUG_TYPE "kit-serialize-tapir-loops"

using namespace llvm;

/// Base class to serialize tapir loops.
class SerializeTapirLoopsImpl {
protected:
  LoopInfo &li;
  OptimizationRemarkEmitter &ore;
  ScalarEvolution &se;
  TaskInfo &ti;

protected:
  /// Serialize the given task. This removes the detach and
  /// corresponding reattach instructions but leaves the syncregion unchanged.
  /// Always returns true.
  bool serializeTask(Task &task) {
    DetachInst *detach = task.getDetach();
    ore.emit(OptimizationRemark(DEBUG_TYPE, "SerializeTapirLoops", detach)
             << "Serializing tapir loop.");
    SerializeDetach(detach, &task);
    return true;
  }

  /// If the given syncregion has only a single use, and the user is a sync
  /// instruction, remove both the sync instruction and the syncregion. Returns
  /// true if the syncregion was removed, false otherwise.
  bool removeSyncRegionAndSync(Value *syncRegion) {
    bool changed = false;
    if (syncRegion->hasOneUse()) {
      User *user = syncRegion->use_begin()->getUser();
      if (auto *syncInst = dyn_cast<SyncInst>(user)) {
        assert(syncInst->getNumSuccessors() == 1 &&
               "Sync instruction must have a single successor");
        BasicBlock *succ = syncInst->getSuccessor(0);
        BranchInst::Create(succ, syncInst->getIterator());
        syncInst->eraseFromParent();

        assert(isa<CallBase>(syncRegion) &&
               "syncregion in detach instruction must be a call");
        cast<CallBase>(syncRegion)->eraseFromParent();

        changed |= true;
      }
    }
    return changed;
  }

  // Strip the tapir loop metadata from the given loop. Always returns true.
  bool stripTapirLoopMetadata(Loop &loop) {
    LLVMContext &ctx = loop.getHeader()->getContext();
    MDNode *loopMD = loop.getLoopID();
    MDNode *newLoopMD =
        makePostTransformationMetadata(ctx, loopMD, {loopMDNamePrefix}, {});
    loop.setLoopID(newLoopMD);

    return true;
  }

  /// Serialize the given tapir loop. The task must be the tapir task
  /// corresponding to the tapir loop. Always returns true.
  bool serializeLoop(Loop &loop, Task &task) {
    bool changed = false;
    DetachInst *detach = task.getDetach();
    Value *syncRegion = detach->getSyncRegion();

    changed |= serializeTask(task);
    changed |= removeSyncRegionAndSync(syncRegion);
    changed |= stripTapirLoopMetadata(loop);

    return true;
  }

  SerializeTapirLoopsImpl(LoopInfo &li, OptimizationRemarkEmitter &ore,
                          ScalarEvolution &se, TaskInfo &ti)
      : li(li), ore(ore), se(se), ti(ti) {}
};

/// Serialize certain tapir loops when the primary tapir target is one of the
/// GPU-centric tapir targets, 'cuda' and 'hip'. Currently, a tapir loop is
/// serialized if:
///
///   a) It is perfectly nested within a tapir loop nest at a level greater than
///      3.
///
///        OR
///
///   b) The loop is part of a tapir loop nest, but is not perfectly nested.
///
/// In each of these cases, the tapir loop will not contain a "perfect.level"
/// annotation.
///
class SerializeTapirLoopsGPU : public SerializeTapirLoopsImpl {
protected:
  /// Reset the "perfect.depth" annotation on the root of the tapir loop nest
  /// to which the given loop belongs. Always returns true.
  bool resetMaxPerfectDepth(Loop &loop, unsigned newDepth) {
    Loop *curr = &loop;
    while (getTapirLoopPerfectLevelMD(*curr) > 1) {
      curr = curr->getParentLoop();
      assert(curr && "Perfectly nested tapir loop at level greater than one "
                     "must have a parent");
    }
    setTapirLoopPerfectDepthMD(*curr, newDepth);
    return true;
  }

public:
  SerializeTapirLoopsGPU(LoopInfo &li, OptimizationRemarkEmitter &ore,
                         ScalarEvolution &se, TaskInfo &ti)
      : SerializeTapirLoopsImpl(li, ore, se, ti) {}

  bool run(Function &f) {
    bool changed = false;
    for (Loop *loop : li.getLoopsInPreorder()) {
      if (Task *task = getTaskIfTapirLoop(loop, &ti)) {
        unsigned perfectLevel = getTapirLoopPerfectLevelMD(*loop);
        if (perfectLevel == 0) {
          // In cases such as those shown below, the innermost forall loops will
          // have not contain a perfect level annotation.
          //
          //   forall (i ...)
          //     for (j ...)
          //       forall (k ...)
          //
          // In such cases, getTapirLoopPerfectLevelMD() will return 0.
          changed |= serializeLoop(*loop, *task);
        } else if (perfectLevel > 3) {
          // When compiling for NVIDIA and AMD GPU's, Kitsune will try to use
          // multidimensional kernel launches when perfect tapir loop nests
          // are found. At the moment, this can be along 3 dimensions at most.
          // Therefore, if deeper perfectly nested tapir loops are found,
          // just serialize them since there is not much we can do about it
          // anyway. In this case, we will also have to adjust the depth
          // annotation at the root of the loop nest.
          changed |= serializeLoop(*loop, *task);

          // We start looking at the parent because loop itself will no longer
          // contain any tapir loop annotations. These are required by
          // resetMaxPerfectDepth().
          changed |= resetMaxPerfectDepth(*loop->getParentLoop(), 3);
        }
      }
    }
    return changed;
  }
};

/// Serialize certain tapir loops when the primary tapir target is 'pthreads'.
class SerializeTapirLoopsPthreads : public SerializeTapirLoopsImpl {
public:
  SerializeTapirLoopsPthreads(LoopInfo &li, OptimizationRemarkEmitter &ore,
                              ScalarEvolution &se, TaskInfo &ti)
      : SerializeTapirLoopsImpl(li, ore, se, ti) {}

  bool run(Function &f) {
    // We don't serialize any tapir loops with the pthreads tapir target,
    // though it is likely that doing so would be profitable. It is for that
    // reason that we create this stub implementation. Otherwise, we could
    // have just returned false from serializeTapirLoops().
    //
    // TODO: Do a performance analysis and either serialize loops in certain
    // cases here, or remove this class altogether and simply return false from
    // serializeTapirLoops().
    return false;
  }
};

/// Serialize certain tapir loops when the primary tapir target is 'qthreads'.
class SerializeTapirLoopsQthreads : public SerializeTapirLoopsImpl {
public:
  SerializeTapirLoopsQthreads(LoopInfo &li, OptimizationRemarkEmitter &ore,
                              ScalarEvolution &se, TaskInfo &ti)
      : SerializeTapirLoopsImpl(li, ore, se, ti) {}

  bool run(Function &f) {
    // We don't serialize any tapir loops with the qthreads tapir target,
    // though it is likely that doing so would be profitable. It is for that
    // reason that we create this stub implementation. Otherwise, we could
    // have just returned false from serializeTapirLoops().
    //
    // TODO: Do a performance analysis and either serialize loops in certain
    // cases here, or remove this class altogether and simply return false from
    // serializeTapirLoops().
    return false;
  }
};

static bool serializeTapirLoops(Function &f, TTID tt, LoopInfo &li,
                                OptimizationRemarkEmitter &ore,
                                ScalarEvolution &se, TaskInfo &ti) {
  switch (tt) {
  case TTID::Nolo:
  case TTID::Serial:
    return false;
  case TTID::Cuda:
  case TTID::Hip:
    // Currently, we use the same logic when serializing tapir loops when
    // using both the 'cuda' and 'hip' tapir targets. If that changes, this
    // will have to change.
    return SerializeTapirLoopsGPU(li, ore, se, ti).run(f);
  case TTID::OpenCilk:
    // We don't currently serialize any tapir loops when using the opencilk
    // tapir target since OpenCilk's runtime is capable of handling nested
    // tapir loops.
    return false;
  case TTID::Pthreads:
    return SerializeTapirLoopsPthreads(li, ore, se, ti).run(f);
  case TTID::Qthreads:
    return SerializeTapirLoopsQthreads(li, ore, se, ti).run(f);
  case TTID::Custom:
    // In principle, the custom tapir target is responsible for handling
    // everything related to lowering, so it is up to the tapir target to
    // handle nested parallel loops correctly.
    //
    // That said, it may be good to have a callback, or some other hook, that
    // could be defined by the tapir target plugin and used here.
    return false;
  case TTID::Realm:
  case TTID::Lambda:
  case TTID::OMPTask:
  case TTID::OpenMP:
    break;
  }
  llvm_unreachable("serializeTapirLoops: TTID not handled");
}

PreservedAnalyses SerializeTapirLoopsPass::run(Module &m,
                                               ModuleAnalysisManager &mam) {
  FunctionAnalysisManager &fam =
      mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();
  TapirTargetInfo &tgi = mam.getResult<TapirTargetAnalysis>(m);
  bool changed = false;

  // This pass currently determines the tapir loops to be serialized based
  // exclusively on the primary tapir target. In the future, we may want to do
  // something more sophisticated, especially when multi-target execution is
  // supported. In that case, we may have to change this implementation.
  for (Function &f : m) {
    if (!f.size())
      continue;

    LoopInfo &li = fam.getResult<LoopAnalysis>(f);
    OptimizationRemarkEmitter &ore =
        fam.getResult<OptimizationRemarkEmitterAnalysis>(f);
    ScalarEvolution &se = fam.getResult<ScalarEvolutionAnalysis>(f);
    TaskInfo &ti = fam.getResult<TaskAnalysis>(f);

    changed |= serializeTapirLoops(f, tgi.getTTID(), li, ore, se, ti);
  }

  if (changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
