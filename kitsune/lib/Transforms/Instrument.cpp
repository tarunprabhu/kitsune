//===- Instrument.cpp - Pass to insert Kitsune's instrumentation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass that inserts Kitsune-specific instrumentation.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/Instrument.h"
#include "kitsune/Core/BasicBlockUtils.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/LibFuncs.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "kit-instrument"

using namespace llvm;

static cl::opt<bool> clKitInstrDumpOpts(
    "kit-instr-dump-opts", cl::init(false),
    cl::desc("Dump Kitsune-specific instrumentation options"), cl::Hidden,
    cl::cat(cl::catKitClOpts));

namespace {

class InstrumentImpl {
private:
  const KitInstrOptions instrOpts;
  LoopInfo &li;
  DomTreeUpdater dtu;
  MemorySSAUpdater mssau;

private:
  Value *addStart(InstrumentKind kind, StringRef name, InsertPosition pt,
                  std::optional<TTID> tt = std::nullopt);
  void addStop(InstrumentKind kind, Value *epoch, InsertPosition pt);
  bool addInstrumentation(StringRef name, InsertPosition startPt,
                          InsertPosition endPt,
                          std::optional<TTID> tt = std::nullopt);
  bool instrumentLoop(Loop &loop);
  bool instrumentLoops(Function &f);
  bool instrumentThreads(Loop &loop);
  bool instrumentThreads(Function &f);
  bool instrumentUnit(InstrumentUnit unit, Function &f);

public:
  InstrumentImpl(const KitInstrOptions &instrOpts, DominatorTree &dt,
                 LoopInfo &li, MemorySSA &mssa)
      : instrOpts(instrOpts), li(li),
        dtu(dt, DomTreeUpdater::UpdateStrategy::Eager), mssau(&mssa) {}

  bool run(Function &f);
};

} // namespace

static Module &getModule(InsertPosition pt) {
  return *getModule(*pt.getBasicBlock());
}

Value *InstrumentImpl::addStart(InstrumentKind kind, StringRef name,
                                InsertPosition pt, std::optional<TTID> tt) {
  auto getKitFunc = [](InstrumentKind kind) -> KitFunc {
    // clang-format off
    switch (kind) {
    case InstrumentKind::Generic: return KitFunc::kit_instr_start;
    case InstrumentKind::PAPI: return KitFunc::kitpapi_start;
    case InstrumentKind::Timer: return KitFunc::kittimer_start;
    }
    // clang-format on
    llvm_unreachable("Instrument::addStart: InstrumentKind not handled");
  };

  Module &m = getModule(pt);
  LLVMContext &ctx = m.getContext();

  SmallVector<Value *, 2> args = {createConstString(name, m)};
  if (tt) {
    Value *ctt = toConstant(*tt, ctx);
    Function *thrdIDFn =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_cpu_thread_id, {});
    Value *thrdID = CallInst::Create(thrdIDFn, ctt, "", pt);
    args.push_back(thrdID);
  } else {
    args.push_back(toConstant(0L, ctx));
  }

  if (kind == InstrumentKind::PAPI) {
    // TODO: Add PAPI counters to measure here.
  }

  FunctionCallee startFn = getOrInsertLibFunc(m, getKitFunc(kind));
  Value *epoch = CallInst::Create(startFn, args, "", pt);

  return epoch;
}

void InstrumentImpl::addStop(InstrumentKind kind, Value *epoch,
                             InsertPosition pt) {
  auto getKitFunc = [](InstrumentKind kind) -> KitFunc {
    // clang-format off
    switch (kind) {
    case InstrumentKind::Generic: return KitFunc::kit_instr_stop;
    case InstrumentKind::PAPI: return KitFunc::kitpapi_stop;
    case InstrumentKind::Timer: return KitFunc::kittimer_stop;
    }
    // clang-format on
    llvm_unreachable("Instrument::addStop: InstrumentKind not handled");
  };

  Module &m = getModule(pt);
  FunctionCallee stopFn = getOrInsertLibFunc(m, getKitFunc(kind));
  (void)CallInst::Create(stopFn, epoch, "", pt);
}

bool InstrumentImpl::addInstrumentation(StringRef name, InsertPosition startPt,
                                        InsertPosition endPt,
                                        std::optional<TTID> tt) {
  SmallVector<InstrumentKind, 1> kinds = instrOpts.getKinds();
  SmallVector<Value *, 1> epochs(kinds.size(), nullptr);

  for (unsigned i = 0, e = epochs.size(); i < e; ++i)
    epochs[i] = addStart(kinds[i], name, startPt, tt);

  for (unsigned i = epochs.size(); i > 0; --i)
    addStop(kinds[i - 1], epochs[i - 1], endPt);

  // If we get this far, we have at least one kind since instrumentation has
  // been enabled, so something will have changed.
  return true;
}

// Add instrumentation around a tapir loop. Consider a loop as shown below. Here
// we have only shown the loop guard, and the exit and end blocks.
//
//     guard:
//       %cmp.n = icmp eq i64 %trip.count, 0
//       br i1 %cmp.n, label %end, label %loop
//
//     loop: ...
//     exit:
//       br label %end
//
//     end:
//       sync within <sync-region>, label %post
//
//     post:
//
// This will be transformed as shown:
//
//     guard:
//       %cmp.n = icmp ...
//       br instr-start:
//
//     instr-start:
//       %epoch = call <instrument-start>("loop-name", 0)
//       br i1 %cmp.n, label %end, label %loop
//
//     loop:
//     exit:
//       br label %end
//
//     end:
//       sync within <sync-region>, label %instr-stop
//
//     instr-stop:
//       call <instrument-stop>(ptr %epoch)
//       br label %post
//
//     post:
//
// Not that instrumentation is added before the guard and after the sync. A
// sync is expected to be found for the loop. If one isn't, found the loop will
// not be instrumented. If the loop is not guarded, the instrumentation start
// functions will be added to the lopo preheader.
bool InstrumentImpl::instrumentLoop(Loop &loop) {
  assert(loop.isLoopSimplifyForm() && "Loop must be in loop-simplify form");

  auto getBlockToInsertStart = [](Loop &loop) -> BasicBlock * {
    // If the loop is guarded, the guard instruction will typically branch to
    // the block containing the sync, which will be a successor of the loop
    // exit block. This is because the loop is in simplify form and is required
    // to have dedicated exits.
    if (BranchInst *br = loop.getLoopGuardBranch())
      return br->getParent();
    else
      return loop.getLoopPreheader();
  };

  // We don't strictly need to split the block in which to insert the
  // instrumentation, but we do so for consistency, and because it makes testing
  // this pass in isolation a shade more reliable.
  BasicBlock *bbBefore = getBlockToInsertStart(loop);
  BasicBlock *bbStart =
      SplitBlock(bbBefore, bbBefore->getTerminator(), &dtu, &li, &mssau,
                 "kit.instr.start", /*Before=*/false);
  InsertPosition startPt = bbStart->getTerminator()->getIterator();

  SyncInst *syncInst = getTapirLoopUniqueSyncInst(loop);
  BasicBlock *bbAfter = syncInst->getSuccessor(0);
  BasicBlock *bbEnd = SplitBlock(bbAfter, bbAfter->begin(), &dtu, &li, &mssau,
                                 "kit.instr.stop", /*Before=*/true);
  InsertPosition endPt = bbEnd->begin();

  StringRef name = getName(loop);
  bool changed = addInstrumentation(name, startPt, endPt);

  return changed;
}

bool InstrumentImpl::instrumentLoops(Function &f) {
  bool changed = false;
  for (Loop *loop : li.getLoopsInPreorder()) {
    StringRef name = getName(*loop);
    if (!isTapirLoop(*loop) || !instrOpts.shouldInstrument(name)) {
      continue;
    } else if (!isTopLevelTapirLoop(*loop)) {
      emitDiagnostic(*loop, DiagID::WarnInstrumentNestedLoop);
      continue;
    }

    SyncInst *syncInst = getTapirLoopUniqueSyncInst(*loop);
    if (!syncInst) {
      emitDiagnostic(*loop, DiagID::WarnInstrumentNoSync);
      continue;
    }

    changed |= instrumentLoop(*loop);
  }
  return changed;
}

// Add instrumentation to run on each thread on which the iterations of a tapir
// loop will be executed. Consider a loop as shown below.
//
//     header:
//       ...
//       detach within <sync-region>, label %body
//
//     body:
//       ...
//       reattach within <sync-region>, label %latch
//
//     latch:
//
// This will be transformed as shown:
//
//     header:
//       ...
//       detach within <sync-region>, label %instr-start
//
//     instr-start:
//       %thrd = call i64 @llvm.kit.cpu.thread.id(i32 <TTID>)
//       %epoch = call <instrument-start>("loop-name", i64 %thrd)
//       br label %body
//
//     body:
//       ...
//       br label %instr-stop
//
//     instr-stop:
//       call <instrument-stop>(ptr %epoch)
//       reattach within <sync-region>, label %latch
//       br label %post
//
//     latch:
//
bool InstrumentImpl::instrumentThreads(Loop &loop) {
  assert(loop.isLoopSimplifyForm() && "Loop must be in loop-simplify form");
  assert(getTapirLoopDetachInst(loop) &&
         "Terminator of loop header must be a detach instruction");
  assert(getTapirLoopReattachInst(loop) &&
         "Unique predecessor of loop latch must be a reattach instruction");

  // Insert a new block immediately before the first detached block in the
  // tapir loop. The detached block will essentially be the first block that
  // is executed on each thread in CPU-centric tapir loops.
  DetachInst *detach = getTapirLoopDetachInst(loop);
  BasicBlock *bbDetached = detach->getDetached();
  BasicBlock *bbStart = SplitBlock(bbDetached, bbDetached->begin(), &dtu, &li,
                                   &mssau, "kit.instr.start", /*Before=*/true);
  InsertPosition startPt = bbStart->begin();

  // Split the block immediately before the loop latch. The terminator of this
  // block will be the tapir loop's reattach instruction. The reattach
  // instruction can be thought of as happening immediately after all threads
  // spawned by a CPU-centric parallel runtime have joined. This way, the
  // instrumentation call will effectively be the last thing that happens in
  // each thread.
  ReattachInst *reattach = getTapirLoopReattachInst(loop);
  BasicBlock *bbReattach = reattach->getParent();
  BasicBlock *bbEnd = SplitBlock(bbReattach, reattach, &dtu, &li, &mssau,
                                 "kit.instr.stop", /*Before=*/false);
  InsertPosition endPt = bbEnd->begin();

  StringRef name = getName(loop);
  TTID tt = *getTargetAttr(loop);
  bool changed = addInstrumentation(name, startPt, endPt, tt);

  return changed;
}

bool InstrumentImpl::instrumentThreads(Function &f) {
  bool changed = false;
  for (Loop *loop : li.getLoopsInPreorder()) {
    StringRef name = getName(*loop);
    if (!isTapirLoop(*loop) || !instrOpts.shouldInstrument(name)) {
      continue;
    } else if (isTopLevelTapirLoopForGPU(*loop)) {
      emitDiagnostic(*loop, DiagID::WarnInstrumentThreadsGPU);
      continue;
    } else if (!isTopLevelTapirLoop(*loop)) {
      emitDiagnostic(*loop, DiagID::WarnInstrumentNestedLoop);
      continue;
    } else if (getTargetAttr(*loop) == TTID::Serial) {
      emitDiagnostic(*loop, DiagID::WarnInstrumentThreadsSerial);
      continue;
    }
    changed |= instrumentThreads(*loop);
  }
  return changed;
}

bool InstrumentImpl::instrumentUnit(InstrumentUnit unit, Function &f) {
  switch (unit) {
  case InstrumentUnit::Loop:
    return instrumentLoops(f);
  case InstrumentUnit::Thread:
    return instrumentThreads(f);
  }
  llvm_unreachable("Instrument::units: Unit not handled");
}

bool InstrumentImpl::run(Function &f) {
  bool changed = false;
  for (InstrumentUnit unit : instrOpts.getUnits())
    changed |= instrumentUnit(unit, f);
  return changed;
}

PreservedAnalyses InstrumentPass::run(Module &m, ModuleAnalysisManager &mam) {
  if (clKitInstrDumpOpts) {
    instrOpts.print(outs());
    return PreservedAnalyses::all();
  }

  FunctionAnalysisManager &fam =
      mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();

  bool changed = false;
  for (Function &f : m) {
    if (f.size()) {
      DominatorTree &dt = fam.getResult<DominatorTreeAnalysis>(f);
      LoopInfo &li = fam.getResult<LoopAnalysis>(f);
      MemorySSA &mssa = fam.getResult<MemorySSAAnalysis>(f).getMSSA();

      changed |= InstrumentImpl(instrOpts, dt, li, mssa).run(f);
    }
  }

  if (changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
