//===----- BlockingPrefetch.cpp - Block loop and issue prefetches ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Carries out an unconditional blocking prefetch transformation. This will
// block a Tapir loop and at iteration i, issue prefetches for data that will
// be used in iteration i + k.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/LoopBlockingPrefetchPass.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopObjectsAnalysis.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Tapir.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "loop-blocking-prefetch"
static const char *passName = DEBUG_TYPE;

using namespace llvm;

cl::opt<bool> llvm::EnableTapirLoopBlockingPrefetch(
    "block-and-prefetch-loops", cl::value_desc("enable"), cl::init(false),
    cl::Hidden,
    cl::desc("Run the Kitsune blocking prefetch transform on Tapir loops"));

cl::opt<unsigned> llvm::TapirBlockingPrefetchBlockCount(
    "block-and-prefetch-num-blocks", cl::value_desc("blocks"), cl::init(0),
    cl::Hidden,
    cl::desc("The number of blocks into which to divide each Tapir loop. This "
             "will effectively be the number of prefetch calls issued for each "
             "Tapir loop."));

// This class will determine the profitability of the blocking prefetch
// transformation on a given loop. This may need to take into account the
// context in which the loop is running i.e. any sibling and/or parent loops and
// perhaps even the larger function. It may also need target information. Since
// the whole point of prefetching is to ovelap computation and data movement, it
// would only make sense to do this in on a target/input where the data movement
// cost is likely to be significant and there is a reasonable expectation that
// overlapping it with the computation can hide that cost. Also, depending on
// how the data is being accessed, it may make sense in some cases to prefetch
// data in an outer loop and in other cases, to prefetch in an inner loop.
class ProfitabilityChecker {
private:
  // TODO: Add any analyses here that might be useful.
  Loop &theLoop;
  Function &func;

public:
  ProfitabilityChecker(Loop &theLoop, Function &func);
  bool run();
};

ProfitabilityChecker::ProfitabilityChecker(Loop &theLoop, Function &func)
    : theLoop(theLoop), func(func) {
  ;
}

bool ProfitabilityChecker::run() {
  // TODO: Actually check the profitability instead of just always assuming
  // that it will be.
  return true;
}

// This class will determine the values to be used for the iteration per
// block (IPB) and the prefetch distance (definitions in the documentation
// for the LoopBlockingPrefetchImpl) for a given loop unless they were
// explicitly overridden.
//
// The values of IPB and distance calcualted by this class may be zero
// which means that there is no way to block the loop or issue a prefetch that
// will ever be profitable. This is deliberately designed this way to allow the
// ProfitabilityChecker to use this class since the analyses needed for
// profitability may be very similar to that needed to calculate the
// parameters.
class ParameterCalculator {
private:
  // TODO: Add any analyses here that might be useful.
  Loop &theLoop;

  // The number of blocks into which to divide the loop.
  unsigned blocks;

private:
  // The default number of blocks into which to divide each Tapir loop.
  // This will be the number of prefetch calls issued for each loop.
  static constexpr unsigned BlockCount = 8;

private:
  unsigned calculateNumBlocks();

public:
  ParameterCalculator(Loop &theLoop);

  // Calculate the parameters. Returns true if valid values for both parameters
  // were obtained, false otherwise.
  bool calculate();

  // Check if the calculated parameters are valid. For this, both ipb and
  // distance must be non-zero.
  bool valid() const;

  unsigned getNumBlocks() const;
  Constant *getNumBlocks(Type *ty) const;
};

ParameterCalculator::ParameterCalculator(Loop &theLoop)
    : theLoop(theLoop), blocks(0) {
  ;
}

unsigned ParameterCalculator::calculateNumBlocks() {
  if (TapirBlockingPrefetchBlockCount)
    return TapirBlockingPrefetchBlockCount;

  // TODO: Actually calculate the iterations per block in a sane way.
  return BlockCount;
}

bool ParameterCalculator::calculate() {
  blocks = calculateNumBlocks();

  return valid();
}

bool ParameterCalculator::valid() const { return blocks; }

unsigned ParameterCalculator::getNumBlocks() const { return blocks; }

Constant *ParameterCalculator::getNumBlocks(Type *ty) const {
  return ConstantInt::get(ty, blocks);
}

// Get the name of the basic block if it exists, otherwise, just return the
// physical address converted to a string. The idea for the latter is that it
// will at least be unique. This is meant purely as a debugging aid anyway.
static std::string getBasicBlockName(BasicBlock &bb) {
  if (bb.hasName())
    return bb.getName().str();

  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << "0x" << &bb;
  return ss.str();
}

// Look for a sync instruction in the given basic block. If it exists, return
// the block, otherwise, expect the basic block to have a single successor and
// check the successor.
//
// While there can be code between the end of the loop and the sync instruction,
// it is expected to be linear. This is a limitation that can be removed but
// is here for simplicity's sake for the moment.
static BasicBlock *getSyncStopBlock(BasicBlock *bb) {
  if (auto *sync = dyn_cast<SyncInst>(bb->getTerminator()))
    return sync->getParent();
  else if (BasicBlock *succ = bb->getSingleSuccessor())
    return getSyncStopBlock(succ);
  return nullptr;
}

// Retrieve the underlying value from a SCEV, if possible.
static Value *getSCEVValue(const SCEV *sc) {
  if (auto *scConst = dyn_cast<SCEVConstant>(sc))
    return scConst->getValue();
  else if (auto *scUnknown = dyn_cast<SCEVUnknown>(sc))
    return scUnknown->getValue();
  return nullptr;
}

// Retrieve the underlying constant value from a SCEV, if possible.
static Value *getSCEVConstant(const SCEV *sc) {
  if (auto *scConst = dyn_cast<SCEVConstant>(sc))
    return scConst->getValue();
  return nullptr;
}

// Check if the value is an integer 0.
static bool isIntZero(Value *v) {
  if (auto *c = dyn_cast<ConstantInt>(v))
    return c->isZero();
  return false;
}

// Check if the value is an integer 1.
static bool isIntOne(Value *v) {
  if (auto *c = dyn_cast<ConstantInt>(v))
    return c->isOne();
  return false;
}

// Emit a missed analysis message with the given emitter if one is available.
static void emitMissedAnalysis(StringRef label, StringRef msg, Loop &loop,
                               OptimizationRemarkEmitter *optRemarks) {
  LLVM_DEBUG(dbgs() << msg << "\n");
  if (optRemarks)
    optRemarks->emit(TapirLoopInfo::createMissedAnalysis(passName, label, &loop)
                     << msg);
}

// Get the primary induction variable of the loop. This need not be canonical.
// This is similar to the corresponding method in TapirLoopInfo, but is more
// generous in the range of valus that the induction variable is allowed to
// span.
static PHINode *
getPrimaryInductionVariable(Loop &loop, PredicatedScalarEvolution &predicatedSE,
                            OptimizationRemarkEmitter *optRemarks = nullptr) {
  // We are only concerned here with finding the loop counter. That is going
  // to be in the header. The loop will already have been transformed into a
  // form that Tapir needs.
  for (Instruction &inst : *loop.getHeader()) {
    // This code is the same as that found in TapirLoopInfo::collectIVs.
    if (auto *phi = dyn_cast<PHINode>(&inst)) {
      Type *phiTy = phi->getType();

      // The forall loop as it is written is (nearly always) not actually
      // parallel. The increment of the loop counter - if present - is a
      // loop-carried dependence which is an obvious violation of the parallel
      // semantics required of a forall. The way we deal with it for all
      // Tapir-enabled optimizations, is to ignore the loop-carried depedence
      // introduced by any counter(s) in the loop (for now, only a single
      // loop counter is permitted, but it would be good to able to generalize
      // this to allow multiple counters operating in different rates and/or
      // directions).
      //
      // The loop header, therefore, ought to only contain the loop counter.
      // If there are any other induction variables here, and of non-integral
      // types, it implies a loop-carried dependence that is unexpected. It
      // could be a straight-up programmer error (which ought to have been
      // caught in Sema, but for some reason wasn't), or it could be that the
      // semantics of forall have changed to allow for more parallel behavior
      // in which case, this function would need to be modified.
      //
      if (not phiTy->isIntegerTy()) {
        emitMissedAnalysis("CFGNotUnderstood",
                           "Loop induction variable must be an integer.", loop,
                           optRemarks);
        return nullptr;
      }

      // Since this is loop header, the PHI node should only have two incoming
      // values - one from outside the loop entering it and the other from the
      // backedge following the loop counter increment and termination test.
      // Anything else suggests that the loop has a more complicated structure.
      // It doesn't automatically mean that the semantics of the forall have
      // been violated, but it is a wrinkle that would need to be analyzed.
      // Currently, we won't do that - not least because any conditional
      // in the loop would make it harder to get a good estimate of a prefetch
      // distance, so that is something that will be punted on for now.
      if (phi->getNumIncomingValues() != 2) {
        emitMissedAnalysis(
            "CFGNotUnderstood",
            "Loop must have exactly 1 backedge and 1 predecessor.", loop,
            optRemarks);
        return nullptr;
      }

      InductionDescriptor indDescr;

      // If the phi can be recognized as an induction variable, then we're done.
      if (InductionDescriptor::isInductionPHI(phi, &loop, predicatedSE,
                                              indDescr))
        return phi;

      // If the phi cannot be recognized as an induction variable, try to
      // coerce it to an AddRec expression and try that way. Not sure why we
      // need to do the latter, but that's what is done in the strip mining
      // pass.
      if (InductionDescriptor::isInductionPHI(phi, &loop, predicatedSE,
                                              indDescr, true))
        return phi;
    }
  }

  return nullptr;
}

class LoopData;

// This struct wraps parameters of a loop induction variable.
class InductionVariable {
private:
  // The induction variable of the loop. In principle, this need not be
  // canonical, but currently Tapir requires a canonical induction variable.
  PHINode *phi = nullptr;

  // The initial value of the loop induction variable. Because of the current
  // limitations of Tapir, this will always be zero. But it is left as a Value*
  // and not a Constant* in case that restriction is ever relaxed.
  Value *init = nullptr;

  // The final value of the loop induction variable.
  Value *finl = nullptr;

  // The per-iteration increment of the loop induction variable. Because of
  // Tapir's current limitations, this is guaranteed to be one. But it is left
  // as a Value* instead of a Constant* in case that restriction is ever
  // relaxed.
  Value *step = nullptr;

public:
  friend class LoopData;
};

// This struct contains the basic blocks on the loops that are modified/created
// in this pass. While adding new loops, the LoopInfo and Loop objects are in
// an inconsistent state. Trying to keep those objects consistent enough that
// they can be queried for the loop preheader etc. requires too much of the
// other code to be reworked. So just look for the blocks before anything is
// changed and use them as required.
//
class LoopData {
private:
  // The loop object.
  Loop *loop = nullptr;

public:
  // The loop's induction variable.
  InductionVariable iv;

  // The single predecessor of the preheader - if it exists.
  BasicBlock *pred = nullptr;

  // The loop's preheader. This is a block outside the loop with a single
  // successor and that successor is the header of the loop. This is guaranteed
  // to be present for any loop that will be modified/created in this pass..
  BasicBlock *preheader = nullptr;

  // The loop header.
  BasicBlock *header = nullptr;

  // The body must have a single predecessor which is the loop header. It
  // must also have a single successor. Not all loops will have a body that
  // satisfies these criteria.
  BasicBlock *body = nullptr;

  // The sole loop latch. This will have two successors - one will be a backedge
  // to the loop header and the other will branch to the exit block. There is
  // guaranteed to be a single latch in any loop that is modified/created in
  // this pass.
  BasicBlock *latch = nullptr;

  // This is the block outside the loop which is one of the destinations of the
  // loop latch. This block will have no other predecessors.
  BasicBlock *exit = nullptr;

  // The single successor of the exit block - if it exists.
  BasicBlock *succ = nullptr;

public:
  LoopData() = default;
  LoopData(Loop &loop, TaskInfo &taskInfo, ScalarEvolution &scEvol,
           PredicatedScalarEvolution &predicatedSE,
           OptimizationRemarkEmitter *optRemarks = nullptr)
      : loop(&loop) {
    preheader = loop.getLoopPreheader();
    header = loop.getHeader();
    latch = loop.getLoopLatch();
    exit = loop.getExitBlock();

    // The check to ensure that this loop is a Tapir loop will already have
    // been performed, so no need to check again.
    Task *tapirTask = getTaskIfTapirLoop(&loop, &taskInfo);
    TapirLoopInfo tapirLoop(&loop, tapirTask);

    if (PHINode *phi =
            getPrimaryInductionVariable(loop, predicatedSE, optRemarks)) {
      // The loop induction variable will probably always need to be an add
      // recurrence, even if the current restrictions on its initial value and
      // step are lifted.
      if (auto *scIV = dyn_cast_or_null<SCEVAddRecExpr>(scEvol.getSCEV(phi))) {
        iv.phi = phi;
        iv.init = getSCEVConstant(scIV->getStart());
        iv.step = getSCEVConstant(scIV->getStepRecurrence(scEvol));

        // The loop induction variable must be finite, but that does not mean
        // the value needs to be known at compile time. It means that the value
        // is loop-invariant and, at runtime, before control enters the loop,
        // the trip count will be known.
        iv.finl =
            tapirLoop.getOrCreateTripCount(predicatedSE, passName, optRemarks);
      }
    }
  }

  void setLoop(Loop &loop) { this->loop = &loop; }

  void setPred(BasicBlock *pred) { this->pred = pred; }
  void setPreheader(BasicBlock *preheader) { this->preheader = preheader; }
  void setHeader(BasicBlock *header) { this->header = header; }
  void setBody(BasicBlock *body) { this->body = body; }
  void setLatch(BasicBlock *latch) { this->latch = latch; }
  void setExit(BasicBlock *exit) { this->exit = exit; }
  void setSucc(BasicBlock *succ) { this->succ = succ; }

  void setIV(PHINode *phi) { iv.phi = phi; }
  void setIVInitial(Value *init) { iv.init = init; }
  void setIVFinal(Value *finl) { iv.finl = finl; }
  void setIVStep(Value *step) { iv.step = step; }

  Loop &getLoop() { return *loop; }

  PHINode *getIV() { return iv.phi; }
  Type *getIVType() { return iv.phi->getType(); }
  Value *getIVInitial() { return iv.init; }
  Value *getIVFinal() { return iv.finl; }
  Value *getIVStep() { return iv.step; }

  BasicBlock *getPred() { return pred; }
  BasicBlock *getPreheader() { return preheader; }
  BasicBlock *getHeader() { return header; }
  BasicBlock *getBody() { return body; }
  BasicBlock *getLatch() { return latch; }
  BasicBlock *getExit() { return exit; }
  BasicBlock *getSucc() { return succ; }
};

// This pass blocks a loop into a size of iterations-per-block (IPB) iterations.
// Before executing the blocked inner loop, at iteration i, the outer loop that
// was added will prefetch some/all of the data that will be used in iteration
// i + distance. The prefetch calls will be made using the asyncprefetch
// intrinsic (which is not the same as the regular prefetch intrinsic that is
// available in standard LLVM). An epilog loop will be added in case the
// loop's trip count is not an integer multiple of IPB.
//
// Currently, this only operates on Tapir loops and therefore, inherits all of
// the limitations that Tapir currently has on the loops on which it will
// operate. Specifically, it requires that the loop induction variable be
// canonical i.e. it must be an integer that starts at 0 and has an increment
// of 1. Some of the descriptions below of the adjustment of the induction
// variables are left here in case Tapir's conditions are relaxed.
//
// Some abbreviations that may be used in the comments in the rest of this
// file:
//
//     PRF       A prefix denoting the prefetch loop i.e. the outer loop that is
//               added as a result of the original loop being blocked. The body
//               of the prefetch loop will contain the prefetch call(s).
//
//     NB        The number of blocks. This will be the same as the number of
//               prefetch calls issued.
//
//     IPB       Iterations per block.
//
//     LTC       Loop trip count (trip count of the loop being transformed).
//
//     OTC       Overflow trip count i.e. the number of "excess" iterations left
//               over as a result of LTC not being an integer multiple of IPB.
//
//     Epilog    The epilog loop is a clone of the original loop but with a
//               trip count of OTC.
//
//
// The loop trip count can be generally calculated as follows:
//
//
//     LTC = floor((IV.end - IV.start) / IV.step)
//
//
// IPB can be determined from NB, and LTC - the number of iterations in the
// loop. Neither NB, nor LTC is required to be a compile-time constant.
//
//
//     IPB = floor(LTC / NB)
//
//
// OTC can be determined from LTC and NB
//
//
//     OTC = LTC % NB
//
//
// where IV is the induction variable of the loop being transformed, IV.start
// is the initial value of IV and IV.step is the per-iteration increment of IV.
//
// The index computations of the blocked loop have to be adjusted as follows.
// For iteration j of the prefetch loop, the initial index of the blocked
// loop will be:
//
//
//     beg = j * IPB * IV.step + IV.start
//
//
// The past-one iteration i.e. the value of the induction variable that will
// terminate execution of the blocked loop at iteration j of the prefetch loop
// will be
//
//
//     end = (j + 1) * IPB * IV.step + IV.start
//         = (j * IPB * IV.step + IV.start) + IPB * IV.step
//         = beg + IPB * IV.step
//
//
// This will convert a loop from the structure shown here:
//
//
//     forall (i = Ns; i < Ne; i += step) {
//       // Some use of a variable.
//       a[i]
//     }
//
//
// to this:
//
//
//      if (LTC / NB != 0) {
//        for (j = 0; j < NB; ++j) {
//          for (i = Ns; i < Ns + IPB * step; i += step) {
//            a[j * IPB * step + Ns + i]
//          }
//        }
//        if (OTC != 0) {
//          for (i = Ns; i < Ns + OTC * step; i += step) {
//            a[NB * IPB * step + Ns + i]
//          }
//        }
//      }
//
//
// The corresponding structure in terms of basic blocks is shown below. The
// first figure is the original loop structure, the second is the structure
// after transformation.
//
//
//      +------<------ loop.predecessor
//      |                   |
//      |              loop.preheader
//      |                   |
//      V              loop.header----<---+
//      |                   |             |
//      |              loop.body          ^
//      |                   |             |
//      |              loop.latch---->----+
//      |                   |
//      +------>------ loop.exit
//
//
// Into something like this:
//
//
//      +------<------ loop.predecessor
//      |                   |
//      |              loop.preheader
//      |                   |
//      |      +---<-- blkprf.predecessor
//      |      |            |
//      |      |       blkprf.preheader
//      |      |            |
//      |      |       blkprf.header -------<-----+
//      |      |            |                     |
//      |      |       blkprf.body                |
//      |      |            |                     |
//      |      V       loop.header --<--+         |
//      |      |            |           |         |
//      |      |       loop.body        ^         |
//      |      |            |           |         ^
//      V      |       loop.latch ------+         |
//      |      |            |                     |
//      |      |       blkprf.latch ------->------+
//      |      |            |
//      |      |       blkprf.exit
//      |      |            |
//      |      +--->-- blkprf.epilog.precheck-->--+
//      |                   |                     |
//      |              epilog.preheader           |
//      |                   |                     |
//      |              epilog.header --<--+       V
//      |                   |             |       |
//      |              epilog.body        ^       |
//      |                   |             |       |
//      |              epilog.latch ------+       |
//      |                   |                     |
//      |              epilog.exit                |
//      |                   |                     |
//      |              blkprf.succ -------<-------+
//      |                   |
//      +------->------loop.sync.block
//
//
// Implementation notes: The loop's preheader, header and exit block are all
// saved before any transformations are initiated. If the LoopInfo object
// and DominatorTree are not updated as the loop nest is being transformed, some
// of the Loop structure query methods such as Loop::getLoopPreheader() and
// Loop::getLoopLatch may return nullptr even if the loop does have a preheader
// (or latch).
class LoopBlockingPrefetchImpl {
private:
  // Loop& theLoop;
  Function &func;
  LLVMContext &ctxt;

  AssumptionCache &assCache;
  DominatorTree &domTree;
  LoopInfo &loopInfo;
  ScalarEvolution &scEvol;
  TaskInfo &taskInfo;
  OptimizationRemarkEmitter *optRemarks;
  PredicatedScalarEvolution predicatedSE;

  // Determines values for IPB and distance if they have not been overridden.
  // If they have, just returns the overridden values. If either of the
  // parameters is invalid, the loop should not be transformed.
  ParameterCalculator paramCalc;

  LoopData theLoop;
  LoopData prfLoop;
  LoopData epilogLoop;

  // The trip count of the main loop (LTC).
  Value *ltc = nullptr;

  // The overflow trip count i.e. the number of iterations left over as a result
  // of the LTC not being an integer multiple of the number of blocks (NB).
  Value *otc = nullptr;

  // Iterations to be performed per block. *(IPB)
  Value *ipb = nullptr;

  // This is used to calculate the start value for the effective loop induction
  // variable for each blocked loop.
  //
  // ipbTimesStep = IPB * IV.step
  Value *ipbTimesStep = nullptr;

private:
  // Replace the successor from the branch at the end of the given basic block
  // from the old successor to the new successor. Returns true if at least one
  // successor of the block was replaced.
  void replaceSuccessorWith(BasicBlock *bb, BasicBlock *oldSucc,
                            BasicBlock *newSucc);

  // Replace the final value of the loop with the new value. The replacement
  // will be done in the compare instruction in the loop latch.
  void replaceLoopIVFinalWith(Loop &loop, Value *oldFinal, Value *newFinal);

  // Replaces all uses of oldVal with newVal except those in the given users.
  void replaceAllUsesWithExcept(Value *oldVal, Value *newVal,
                                const SmallSet<User *, 4> &except);

  // Get the instruction that increments the induction variable for use in the
  // loop latch where it is compared to the given final value.
  User *getLoopIVIncr(Loop &loop, Value *finl);

  // Return true if the blocking prefetch transform can be applied to the loop.
  // This will also initialize some of the loop-specific fields that were
  // set to null in the constructor.
  bool canBeTransformed();

  // Fix the main loop preheader. This adds calculations for the loop trip
  // count (LTC), overflow trip count (OTC) and iterations per block (IPB)
  // which are needed in several different places. Since the loop preheader
  // dominates the main loop, it will also dominate the prefetch loop and the
  // epilog.
  void fixLoopPreheader();

  // Populate prfLoop.
  void populatePrfLoop();

  // Populate the epilog loop.
  void populateEpilogLoop();

  // Create a prefetch loop object in the LoopInfo analysis. This will create
  // the loop, set the children and parents of the loop correctly, add any
  // basic blocks to the loop that are not automatically added.
  void createPrfLoop();

  // Create an epilog loop for cases where the loop trip count is not an
  // exact multiple of the number of iterations in the block.
  void createEpilogLoop();

  // Update the dominator tree.
  void updateDomTree();

  // Update the loop's predecessors, successors, induction variable and latch.
  void fixTheLoop();

  // Update the epilog loop's predecessors, successors and induction variable.
  void fixEpilogLoop();

  // Add a metadata node to the loop's id node to prevent this transformation
  // from running on the loop.
  void addDisableMetadata(Loop &loop);

  // Check if the loop has a metadata node indicating that this transformation
  // has been disabled. This will likely have been added if this transformation
  // has already been applied to the loop or the loop was generated as a result
  // of applying this transformation to another loop.
  bool hasDisableMetadata(Loop &loop);

public:
  LoopBlockingPrefetchImpl(Loop &theLoop, Function &func,
                           AssumptionCache &assCache, DominatorTree &domTree,
                           LoopInfo &loopInfo, ScalarEvolution &scEvol,
                           TaskInfo &taskInfo,
                           OptimizationRemarkEmitter *optRemarks = nullptr);
  bool run();
};

LoopBlockingPrefetchImpl::LoopBlockingPrefetchImpl(
    Loop &loop, Function &func, AssumptionCache &assCache,
    DominatorTree &domTree, LoopInfo &loopInfo, ScalarEvolution &scEvol,
    TaskInfo &taskInfo, OptimizationRemarkEmitter *optRemarks)
    : func(func), ctxt(func.getContext()), assCache(assCache), domTree(domTree),
      loopInfo(loopInfo), scEvol(scEvol), taskInfo(taskInfo),
      optRemarks(optRemarks), predicatedSE(scEvol, loop), paramCalc(loop),
      theLoop(loop, taskInfo, scEvol, predicatedSE, optRemarks) {}

void LoopBlockingPrefetchImpl::replaceSuccessorWith(BasicBlock *bb,
                                                    BasicBlock *oldSucc,
                                                    BasicBlock *newSucc) {
  bool replaced = false;

  if (BranchInst *br = dyn_cast<BranchInst>(bb->getTerminator())) {
    for (unsigned i = 0; i < br->getNumSuccessors(); i++) {
      bool shouldReplace = br->getSuccessor(i) == oldSucc;
      if (shouldReplace)
        br->setSuccessor(i, newSucc);
      replaced |= shouldReplace;
    }
  }

  if (not replaced)
    emitMissedAnalysis("CFGNotUnderstood",
                       "Could not replace successor of basic block.",
                       theLoop.getLoop(), optRemarks);
}

void LoopBlockingPrefetchImpl::replaceLoopIVFinalWith(Loop &loop,
                                                      Value *oldFinal,
                                                      Value *newFinal) {
  BasicBlock *latch = loop.getLoopLatch();

  // Find the comparison instruction and update the termination check there.
  // TODO: At some point, it would be very nice to use replaceAllUsesWithIf
  // here.
  auto *br = dyn_cast<BranchInst>(latch->getTerminator());
  assert(br && "Terminator of the loop latch must be a branch instruction.");

  auto *cmp = dyn_cast<CmpInst>(br->getCondition());
  assert(cmp && "Value of the conditional branch instruction in the loop latch "
                "must be a compare instruction.");

  unsigned idx = cmp->getOperand(0) == oldFinal ? 0 : 1;
  cmp->setOperand(idx, newFinal);
}

void LoopBlockingPrefetchImpl::replaceAllUsesWithExcept(
    Value *oldVal, Value *newVal, const SmallSet<User *, 4> &except) {
  // Collect the uses first and then replace the operands there.
  // TODO: At some point, it would be very nice to use replaceAllUsesWithIf
  // here.
  SmallVector<Use *, 8> uses;
  for (Use &u : oldVal->uses())
    if (not except.contains(u.getUser()))
      uses.push_back(&u);

  for (Use *use : uses) {
    unsigned opNo = use->getOperandNo();
    User *user = use->getUser();
    user->setOperand(opNo, newVal);
  }
}

User *LoopBlockingPrefetchImpl::getLoopIVIncr(Loop &loop, Value *finl) {
  BasicBlock *latch = loop.getLoopLatch();

  auto *br = dyn_cast<BranchInst>(latch->getTerminator());
  assert(br && "Terminator of the loop latch must be a branch instruction.");

  auto *cmp = dyn_cast<CmpInst>(br->getCondition());
  assert(cmp && "Value of the conditional branch instruction in the loop latch "
                "must be a compare instruction.");

  unsigned idx = cmp->getOperand(0) == finl ? 1 : 0;
  User *ivIncr = dyn_cast<User>(cmp->getOperand(idx));
  assert(ivIncr && "Increment of loop induction variable must be a User.");

  return ivIncr;
}

bool LoopBlockingPrefetchImpl::canBeTransformed() {
  Loop &loop = theLoop.getLoop();

  if (hasDisableMetadata(loop)) {
    emitMissedAnalysis("Disabled",
                       "Transformation has been explicitly disabled on loop.",
                       loop, optRemarks);
    return false;
  }

  // The loop must have an induction variable.
  PHINode *iv = theLoop.getIV();
  if (not iv) {
    emitMissedAnalysis("NoPrimaryIndVar",
                       "Loop does not have a primary induction variable.", loop,
                       optRemarks);
    return false;
  }
  LLVM_DEBUG(dbgs() << "Loop induction variable: " << *iv << "\n");

  // The loop induction variable will probably always need to be an add
  // recurrence, even if the current restrictions on its initial value and
  // step are lifted.
  auto *scIV = dyn_cast_or_null<SCEVAddRecExpr>(scEvol.getSCEV(iv));
  if (not scIV) {
    emitMissedAnalysis("NoAddRecIndVar",
                       "Loop induction variable is not an add recurrence.",
                       loop, optRemarks);
    return false;
  }

  // The loop induction variable must be canonical, therefore its initial value
  // must be zero.
  Value *ivInit = theLoop.getIVInitial();
  if (not ivInit) {
    emitMissedAnalysis(
        "NonConstIVInitial",
        "Could not determine finite initial value for loop induction variable",
        loop, optRemarks);
    return false;
  } else if (not isIntZero(ivInit)) {
    emitMissedAnalysis("NonZeroIVInitial",
                       "Loop induction variable is not canonical.", loop,
                       optRemarks);
    return false;
  }
  LLVM_DEBUG(dbgs() << "Loop IV initial: " << *ivInit << "\n");

  // The loop induction variable must be canonical, therefore its increment
  // (step) must be one.
  Value *ivStep = theLoop.getIVStep();
  if (not ivStep) {
    emitMissedAnalysis(
        "NonConstIVStep",
        "Could not determine finite step for loop induction variable.", loop,
        optRemarks);
    return false;
  } else if (not isIntOne(ivStep)) {
    emitMissedAnalysis("NonUnitIVStep",
                       "Loop induction variable step is not 1.", loop,
                       optRemarks);
    return false;
  }
  LLVM_DEBUG(dbgs() << "Loop IV step: " << *ivStep << "\n");

  // Simplify loop will already have been called before we get here, but that
  // function could have failed and the loop may not be in simplified form.
  // So check for everything here.
  BasicBlock *loopPreheader = theLoop.getPreheader();
  if (not loopPreheader) {
    emitMissedAnalysis("CFGNotUnderstood", "Loop does not have a preheader.",
                       loop, optRemarks);
    return false;
  }
  LLVM_DEBUG(dbgs() << "Loop preheader: " << getBasicBlockName(*loopPreheader)
                    << "\n");

  // The loop will always have a header.
  BasicBlock *loopHeader = theLoop.getHeader();
  LLVM_DEBUG(dbgs() << "Loop header: " << getBasicBlockName(*loopHeader)
                    << "\n");

  BasicBlock *loopLatch = theLoop.getLatch();
  if (not loopLatch) {
    emitMissedAnalysis("CFGNotUnderstood", "Loop does not have a unique latch.",
                       loop, optRemarks);
    return false;
  }
  LLVM_DEBUG(dbgs() << "Loop latch: " << getBasicBlockName(*loopLatch) << "\n");

  BasicBlock *loopExit = theLoop.getExit();
  if (not loopExit) {
    emitMissedAnalysis("CFGNotUnderstood",
                       "Loop does not have a unique exit block.", loop,
                       optRemarks);
    return false;
  }
  LLVM_DEBUG(dbgs() << "Loop exit: " << getBasicBlockName(*loopExit) << "\n");

  // calculate() will return false if a non-zero value for either IPB or
  // distance could not be determined.
  if (not paramCalc.calculate()) {
    emitMissedAnalysis(
        "NotProfitable",
        "Could not determine profitable parameters for the transformation.",
        loop, optRemarks);
    return false;
  }
  LLVM_DEBUG(dbgs() << "Number of blocks: " << paramCalc.getNumBlocks()
                    << "\n");

  return true;
}

void LoopBlockingPrefetchImpl::createEpilogLoop() {
  ValueToValueMapTy vmap;
  SmallVector<BasicBlock *, 8> blocks;
  BasicBlock *loopPreheader = theLoop.getPreheader();
  BasicBlock *loopExit = theLoop.getExit();
  PHINode *loopIV = theLoop.getIV();

  BasicBlock *epilogPred =
      BasicBlock::Create(ctxt, "blkprf.epilog.precheck", &func, loopExit);
  BasicBlock *epilogExit =
      BasicBlock::Create(ctxt, "blkprf.epilog.exit", &func, loopExit);
  BasicBlock *epilogSucc =
      BasicBlock::Create(ctxt, "blkprf.epilog.succ", &func, loopExit);

  vmap[loopExit] = epilogExit;
  Loop *epilogLoop =
      cloneLoopWithPreheader(epilogExit, loopPreheader, &theLoop.getLoop(),
                             vmap, ".epilog", &loopInfo, &domTree, blocks);
  remapInstructionsInBlocks(blocks, vmap);

  this->epilogLoop.setLoop(*epilogLoop);

  this->epilogLoop.setPred(epilogPred);
  this->epilogLoop.setPreheader(epilogLoop->getLoopPreheader());
  this->epilogLoop.setHeader(epilogLoop->getHeader());
  this->epilogLoop.setLatch(epilogLoop->getLoopLatch());
  this->epilogLoop.setExit(epilogExit);
  this->epilogLoop.setSucc(epilogSucc);

  // Since the epilog loop is a clone of the main loop, the initial and final
  // values of the induction variable and the step will be identical for both.
  // But we could consider using scalar evolution to recalculate them here. In
  // fact, that would probably be better in the general case since the initial
  // value need not be a constant in general. We can get away with this for now
  // because of Tapir's constraints.
  this->epilogLoop.setIV(cast<PHINode>(vmap[loopIV]));
  this->epilogLoop.setIVInitial(theLoop.getIVInitial());
  this->epilogLoop.setIVFinal(theLoop.getIVFinal());
  this->epilogLoop.setIVStep(theLoop.getIVStep());

  LLVM_DEBUG(dbgs() << "Epilog preheader: "
                    << getBasicBlockName(*this->epilogLoop.getPreheader())
                    << "\n");
  LLVM_DEBUG(dbgs() << "Epilog header: "
                    << getBasicBlockName(*this->epilogLoop.getHeader())
                    << "\n");
  LLVM_DEBUG(dbgs() << "Epilog latch: "
                    << getBasicBlockName(*this->epilogLoop.getLatch()) << "\n");
  LLVM_DEBUG(dbgs() << "Epilog exit: "
                    << getBasicBlockName(*this->epilogLoop.getExit()) << "\n");
}

void LoopBlockingPrefetchImpl::createPrfLoop() {
  BasicBlock *loopHeader = theLoop.getHeader();
  BasicBlock *loopExit = theLoop.getExit();

  BasicBlock *epilogPred = epilogLoop.getPred();
  BasicBlock *epilogExit = epilogLoop.getExit();
  BasicBlock *epilogSucc = epilogLoop.getSucc();

  // Create empty basic blocks for different elements of the prefetch loop.
  // The blocks will be created in roughly the "right" place to make the
  // control-flow reasonably sane when reading the IR, but the blocks will not
  // have any branch instructions. Those will be added by the populate*
  // methods.
  BasicBlock *prfPred =
      BasicBlock::Create(ctxt, "blkprf.precheck", &func, loopHeader);
  BasicBlock *prfPreheader =
      BasicBlock::Create(ctxt, "blkprf.preheader", &func, loopHeader);
  BasicBlock *prfHeader =
      BasicBlock::Create(ctxt, "blkprf.header", &func, loopHeader);
  BasicBlock *prfBody =
      BasicBlock::Create(ctxt, "blkprf.body", &func, loopHeader);
  BasicBlock *prfLatch =
      BasicBlock::Create(ctxt, "blkprf.latch", &func, loopExit);
  BasicBlock *prfExit =
      BasicBlock::Create(ctxt, "blkprf.exit", &func, loopExit);

  Loop *prfLoop = loopInfo.AllocateLoop();

  this->prfLoop.setLoop(*prfLoop);
  this->prfLoop.setPred(prfPred);
  this->prfLoop.setPreheader(prfPreheader);
  this->prfLoop.setHeader(prfHeader);
  this->prfLoop.setBody(prfBody);
  this->prfLoop.setLatch(prfLatch);
  this->prfLoop.setExit(prfExit);
  this->prfLoop.setSucc(epilogPred);

  Loop &theLoop = this->theLoop.getLoop();
  if (Loop *parent = theLoop.getParentLoop()) {
    parent->removeChildLoop(&theLoop);

    parent->addBasicBlockToLoop(prfPred, loopInfo);
    parent->addBasicBlockToLoop(prfPreheader, loopInfo);
    parent->addBasicBlockToLoop(prfExit, loopInfo);
    parent->addBasicBlockToLoop(epilogPred, loopInfo);
    parent->addBasicBlockToLoop(epilogExit, loopInfo);
    parent->addBasicBlockToLoop(epilogSucc, loopInfo);

    parent->addChildLoop(prfLoop);
  } else {
    loopInfo.changeTopLevelLoop(&theLoop, prfLoop);
  }

  prfLoop->addBasicBlockToLoop(prfHeader, loopInfo);
  prfLoop->addBasicBlockToLoop(prfBody, loopInfo);
  for (BasicBlock *bb : theLoop.blocks())
    prfLoop->addBlockEntry(bb);
  prfLoop->addBasicBlockToLoop(prfLatch, loopInfo);

  prfLoop->addChildLoop(&theLoop);
}

void LoopBlockingPrefetchImpl::updateDomTree() {
  BasicBlock *loopPreheader = theLoop.getPreheader();
  BasicBlock *loopHeader = theLoop.getHeader();
  BasicBlock *loopLatch = theLoop.getLatch();
  BasicBlock *loopExit = theLoop.getExit();

  BasicBlock *prfPred = prfLoop.getPred();
  BasicBlock *prfPreheader = prfLoop.getPreheader();
  BasicBlock *prfHeader = prfLoop.getHeader();
  BasicBlock *prfBody = prfLoop.getBody();
  BasicBlock *prfLatch = prfLoop.getLatch();
  BasicBlock *prfExit = prfLoop.getExit();

  BasicBlock *epilogPred = epilogLoop.getPred();
  BasicBlock *epilogPreheader = epilogLoop.getPreheader();
  BasicBlock *epilogLatch = epilogLoop.getLatch();
  BasicBlock *epilogExit = epilogLoop.getExit();
  BasicBlock *epilogSucc = epilogLoop.getSucc();

  // First add all the new blocks. The first argument is the block being added
  // and the second is the dominator of that block.
  domTree.addNewBlock(prfPred, loopPreheader);
  domTree.addNewBlock(prfPreheader, prfPred);
  domTree.addNewBlock(prfHeader, prfPreheader);
  domTree.addNewBlock(prfBody, prfHeader);
  domTree.addNewBlock(prfLatch, loopLatch);
  domTree.addNewBlock(prfExit, prfLatch);
  domTree.addNewBlock(epilogPred, prfPred);
  domTree.addNewBlock(epilogSucc, epilogPred);
  domTree.addNewBlock(epilogExit, epilogLatch);

  // Now change the dominators accordingly.
  domTree.changeImmediateDominator(loopHeader, prfBody);
  domTree.changeImmediateDominator(loopExit, epilogSucc);
  domTree.changeImmediateDominator(epilogPreheader, epilogPred);
}

void LoopBlockingPrefetchImpl::fixLoopPreheader() {
  Type *ivTy = theLoop.getIVType();
  Value *numBlocks = paramCalc.getNumBlocks(ivTy);
  BasicBlock *loopPreheader = theLoop.getPreheader();
  Instruction *preheaderTerm = loopPreheader->getTerminator();
  Value *loopIVInitial = theLoop.getIVInitial();
  Value *loopIVStep = theLoop.getIVStep();
  Value *loopIVFinal = theLoop.getIVFinal();

  // The is an unnecessary general calculation since the loop induction variable
  // is currently required to be canonical. But leave it this way in case that
  // restriction is lifted. Any redundant computations (such as division by 1
  // or addition with 0) will be strength-reduced away.

  // Loop trip count (LTC)
  //
  //     LTC = (IV.stop - IV.start) / IV.step
  //
  // sub = IV.stop - IV.start
  Value *sub = BinaryOperator::Create(Instruction::Sub, loopIVFinal,
                                      loopIVInitial, "", preheaderTerm);

  // LTC = sub / IV.step
  ltc = BinaryOperator::Create(Instruction::UDiv, sub, loopIVStep,
                               "loop.trip.count", preheaderTerm);

  // IPB = LTC / NB
  ipb = BinaryOperator::Create(Instruction::UDiv, ltc, numBlocks, "loop.ipb",
                               preheaderTerm);

  // OTC = ltc % NB
  otc = BinaryOperator::Create(Instruction::URem, ltc, numBlocks,
                               "loop.trip.rem", preheaderTerm);

  // ipbTimesStep = IPB * IV.step
  ipbTimesStep = BinaryOperator::Create(Instruction::Mul, ipb, loopIVStep,
                                        "loop.ipb.x.step", preheaderTerm);
}

void LoopBlockingPrefetchImpl::populatePrfLoop() {
  Type *ivTy = theLoop.getIVType();
  Constant *zero = ConstantInt::get(ivTy, 0);
  Constant *one = ConstantInt::get(ivTy, 1);
  Value *numBlocks = paramCalc.getNumBlocks(ivTy);

  BasicBlock *loopHeader = theLoop.getHeader();

  BasicBlock *prfPred = prfLoop.getPred();
  BasicBlock *prfPreheader = prfLoop.getPreheader();
  BasicBlock *prfHeader = prfLoop.getHeader();
  BasicBlock *prfBody = prfLoop.getBody();
  BasicBlock *prfLatch = prfLoop.getLatch();
  BasicBlock *prfExit = prfLoop.getExit();

  BasicBlock *epilogPred = epilogLoop.getPred();

  // Populate pred
  // if (ltc < NB) goto epilogPredecessor else goto prefetchPreheader
  CmpInst *numBlocksCmp = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_ULT,
                                          ltc, numBlocks, "", prfPred);
  BranchInst::Create(epilogPred, prfPreheader, numBlocksCmp, prfPred);

  // Populate preheader
  BranchInst::Create(prfHeader, prfPreheader);

  // Populate header
  prfLoop.setIV(PHINode::Create(ivTy, 2, "blkprf.iv", prfHeader));
  PHINode *prfIV = prfLoop.getIV();

  Instruction *prfIVIncr = BinaryOperator::CreateNUW(
      Instruction::Add, prfIV, one, "blkprf.iv.next", prfHeader);

  prfIV->addIncoming(zero, prfPreheader);
  prfIV->addIncoming(prfIVIncr, prfLatch);

  BranchInst::Create(prfBody, prfHeader);

  // Populate body
  // FIXME: Populate the body of the prefetch loop.

  // Module *mod = func.getParent();

  // PointerType *i8PtrTy = Type::getInt8PtrTy(ctxt);
  // Type *i64Ty = Type::getInt64Ty(ctxt);
  // Type *i32Ty = Type::getInt32Ty(ctxt);
  // Type *paramTys[] = {i8PtrTy};

  // FunctionType *prfTy =
  //     llvm::Intrinsic::getType(ctxt, Intrinsic::asyncprefetch, paramTys);
  // Function *prfFn = llvm::Intrinsic::getDeclaration(
  //     mod, Intrinsic::asyncprefetch, paramTys);

  // // FIXME: These should be actual arguments
  // Value *ptr = ConstantPointerNull::get(i8PtrTy);
  // Value *size = ConstantInt::get(i64Ty, 0);
  // Value *mode = ConstantInt::get(i32Ty, 0);
  // Value *prfArgs[] = {ptr, size, mode};

  // CallInst::Create(prfTy, prfFn, prfArgs, "", prfBody);

  BranchInst::Create(loopHeader, prfBody);

  // Populate latch.
  // if (iv != NB) goto prfHeader else goto prfExit
  CmpInst *prfVarCmp =
      CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_NE, prfIVIncr, numBlocks,
                      "blkprf.exitcond.not", prfLatch);
  BranchInst::Create(prfHeader, prfExit, prfVarCmp, prfLatch);

  // Populate exit.
  BranchInst::Create(epilogPred, prfExit);
}

// Populate epilog loop
void LoopBlockingPrefetchImpl::populateEpilogLoop() {
  Type *ivTy = theLoop.getIVType();
  Constant *zero = ConstantInt::get(ivTy, 0);

  BasicBlock *loopExit = theLoop.getExit();

  BasicBlock *epilogPred = epilogLoop.getPred();
  BasicBlock *epilogPreheader = epilogLoop.getPreheader();
  BasicBlock *epilogExit = epilogLoop.getExit();
  BasicBlock *epilogSucc = epilogLoop.getSucc();

  // Populate pred
  // if (OTC != 0) goto epilogLoopPreheader else goto epilogLoopSuccessor
  CmpInst *otcCmp = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_NE, otc,
                                    zero, "", epilogPred);
  BranchInst::Create(epilogPreheader, epilogSucc, otcCmp, epilogPred);

  // The blocks comprising the main loop i.e. the header, body, and latch do
  // not need to be changed. The loop bounds will be fixed at the end when
  // fixEpilogLoop() is called.

  // Populate exit.
  BranchInst::Create(epilogSucc, epilogExit);

  // Populate succ.
  BranchInst::Create(loopExit, epilogSucc);
}

void LoopBlockingPrefetchImpl::fixTheLoop() {
  BasicBlock *loopPreheader = theLoop.getPreheader();
  BasicBlock *loopHeader = theLoop.getHeader();
  BasicBlock *loopLatch = theLoop.getLatch();
  BasicBlock *loopExit = theLoop.getExit();

  BasicBlock *prfPred = prfLoop.getPred();
  BasicBlock *prfBody = prfLoop.getBody();
  BasicBlock *prfLatch = prfLoop.getLatch();

  PHINode *prfIV = prfLoop.getIV();
  Instruction *prfBodyTerm = prfBody->getTerminator();

  PHINode *loopIV = theLoop.getIV();
  Value *loopIVInitial = theLoop.getIVInitial();
  Value *loopIVFinal = theLoop.getIVFinal();

  // The effective new initial value of the loop induction variable is needed
  // to adjust the offsets in any array accesses within the loop. The actual
  // values of the induction variable itself are not modified because Tapir
  // currently requires that the induction variable of Tapir loops be
  // canonical.
  //
  //     new-initial = IV.init + prfIV * IPB * IV.step
  //
  // scInit = prfIV * IPB * IV.step
  Value *scInit = BinaryOperator::Create(Instruction::Mul, prfIV, ipbTimesStep,
                                         "", prfBodyTerm);

  Value *newLoopIVInitial = BinaryOperator::Create(
      Instruction::Add, scInit, loopIVInitial, "blk.iv.init", prfBodyTerm);

  // The effective new final value of the loop induction variable on the other
  // hand does have to be used in the latch to ensure that the loop does not
  // run more iterations than we need it to.
  //
  //     new-final = IV.init + IPB * IV.step
  //
  Value *newLoopIVFinal =
      BinaryOperator::Create(Instruction::Add, loopIVInitial, ipbTimesStep,
                             "blk.iv.final", prfBodyTerm);

  replaceLoopIVFinalWith(theLoop.getLoop(), loopIVFinal, newLoopIVFinal);

  // Since the initial value of the loop induction variable has not been
  // changed, all the uses must be updated
  //
  //     EXCEPT
  //
  //   1. the place where the induction variable is incremented (since the
  //      induction variable must be an AddRecExpr, I assume that it will always
  //      be incremented even if it is with a negative number ... hopefully).
  //
  //   2. the place where the new value of the index is computed since that will
  //      also need to use the old value of the induction variable.
  //
  // The new value of the index that should be used will be
  //
  //     new-iv = IV.init + prfIV * IPB * IV.step + IV
  //
  Instruction *succ = loopIV->getNextNonDebugInstruction();
  User *newLoopIV =
      BinaryOperator::Create(Instruction::Add, newLoopIVInitial, loopIV,
                             loopIV->getName() + ".adj", succ);

  SmallSet<User *, 4> except;
  except.insert(newLoopIV);
  except.insert(getLoopIVIncr(theLoop.getLoop(), newLoopIVFinal));
  replaceAllUsesWithExcept(loopIV, newLoopIV, except);

  // Rewire the connections to the loops. This mainly deals with setting up
  // unconditional branches which should take care of ensuring that all the
  // loops can find headers, preheade
  replaceSuccessorWith(loopPreheader, loopHeader, prfPred);
  replaceSuccessorWith(loopLatch, loopExit, prfLatch);

  // The edge that came from the loop's pre-header will now come from the
  // prefetch loop's body. Update the PHI node to reflect this.
  //
  loopIV->replaceIncomingBlockWith(loopPreheader, prfBody);
}

void LoopBlockingPrefetchImpl::fixEpilogLoop() {
  Type *ivTy = theLoop.getIVType();
  Value *numBlocks = paramCalc.getNumBlocks(ivTy);

  BasicBlock *epilogPreheader = epilogLoop.getPreheader();
  Instruction *epilogPreheaderTerm = epilogPreheader->getTerminator();

  PHINode *epilogIV = epilogLoop.getIV();
  Value *epilogIVInitial = epilogLoop.getIVInitial();
  Value *epilogIVFinal = epilogLoop.getIVFinal();
  Value *epilogIVStep = epilogLoop.getIVStep();

  // The effective new initial value of the loop induction variable is needed
  // to adjust the offsets in any array accesses within the loop. The actual
  // values of the induction variable itself are not modified because Tapir
  // currently requires that the induction variable of Tapir loops be
  // canonical.
  //
  //     new-initial = IV.init + NB * IPB * IV.step
  //
  // scInit = NB * IPB * IV.step
  Value *scInit = BinaryOperator::Create(Instruction::Mul, numBlocks,
                                         ipbTimesStep, "", epilogPreheaderTerm);

  Value *newEpilogIVInitial =
      BinaryOperator::Create(Instruction::Add, scInit, epilogIVInitial,
                             "epilog.iv.init", epilogPreheaderTerm);

  // The effective new final value of the loop induction variable on the other
  // hand does have to be used in the latch to ensure that the loop does not
  // run more iterations than we need it to.
  //
  //     new-final = IV.init + OTC * IV.step
  //
  Value *sc = BinaryOperator::Create(Instruction::Mul, otc, epilogIVStep, "",
                                     epilogPreheaderTerm);
  Value *newEpilogIVFinal =
      BinaryOperator::Create(Instruction::Add, epilogIVInitial, sc,
                             "epilog.iv.final", epilogPreheaderTerm);

  replaceLoopIVFinalWith(epilogLoop.getLoop(), epilogIVFinal, newEpilogIVFinal);

  // Since the initial value of the loop induction variable has not been
  // changed, all the uses must be updated
  //
  //     EXCEPT
  //
  //   1. the place where the induction variable is incremented.
  //
  //   2. the place where the new value of the index is computed since that will
  //      also need to use the old value of the induction variable.
  //
  // The new value of the index that should be used will be
  //
  //     new-iv = IV.init + NB * IPB * IV.step + IV
  //
  Instruction *succ = epilogIV->getNextNonDebugInstruction();
  User *newEpilogIV =
      BinaryOperator::Create(Instruction::Add, newEpilogIVInitial, epilogIV,
                             epilogIV->getName() + ".adj", succ);

  SmallSet<User *, 4> except;
  except.insert(newEpilogIV);
  except.insert(getLoopIVIncr(epilogLoop.getLoop(), newEpilogIVFinal));
  replaceAllUsesWithExcept(epilogIV, newEpilogIV, except);
}

void LoopBlockingPrefetchImpl::addDisableMetadata(Loop &loop) {
  Metadata *mds = MDString::get(ctxt, "llvm.loop.block-and-prefetch.disable");
  MDNode *mdn = MDNode::get(ctxt, mds);
  MDNode *newId = makePostTransformationMetadata(
      ctxt, loop.getLoopID(), {"llvm.loop.block-and-prefetch"}, mdn);

  loop.setLoopID(newId);
}

bool LoopBlockingPrefetchImpl::hasDisableMetadata(Loop &loop) {
  return findOptionMDForLoop(&loop, "llvm.loop.block-and-prefetch.disable");
}

bool LoopBlockingPrefetchImpl::run() {
  // This mainly checks if the loop is in the right "shape" to be transformed.
  // Profitability checks are assumed to have been performed already. It will
  // also check if valid values for iterations per block and prefetch distance
  // can be computed (if they have not been overridden).
  if (not canBeTransformed())
    return false;

  // Create the epilog loop before creating the prefetch loop because it will
  // clone the main loop and it is best that it clone a loop that has not been
  // transformed in any way.
  createEpilogLoop();

  // Now, create the prefetch loop. This will hook up the prefetch loop with
  // the parent of the loop (if any), add the right basic blocks to it and
  // the any parent.
  createPrfLoop();

  // Update dominator tree relatively early in the process so if it is needed
  // later during the transformations, it is still in good shape.
  updateDomTree();

  fixLoopPreheader();
  populatePrfLoop();
  populateEpilogLoop();

  fixTheLoop();
  fixEpilogLoop();

  addDisableMetadata(theLoop.getLoop());
  addDisableMetadata(prfLoop.getLoop());
  addDisableMetadata(epilogLoop.getLoop());

  simplifyLoop(&prfLoop.getLoop(), &domTree, &loopInfo, &scEvol, &assCache,
               nullptr, false);

  simplifyLoop(&theLoop.getLoop(), &domTree, &loopInfo, &scEvol, &assCache,
               nullptr, false);

  simplifyLoop(&epilogLoop.getLoop(), &domTree, &loopInfo, &scEvol, &assCache,
               nullptr, false);

  // FIXME: Use object analysis.
  // FIXME: Add prefetch calls.

  return true;
}

PreservedAnalyses LoopBlockingPrefetchPass::run(Function &func,
                                                FunctionAnalysisManager &am) {
  if (not EnableTapirLoopBlockingPrefetch)
    return PreservedAnalyses::all();

  AssumptionCache &assCache = am.getResult<AssumptionAnalysis>(func);
  DominatorTree &domTree = am.getResult<DominatorTreeAnalysis>(func);
  LoopInfo &loopInfo = am.getResult<LoopAnalysis>(func);
  OptimizationRemarkEmitter &optRemarks =
      am.getResult<OptimizationRemarkEmitterAnalysis>(func);
  ScalarEvolution &scEvol = am.getResult<ScalarEvolutionAnalysis>(func);
  TaskInfo &taskInfo = am.getResult<TaskAnalysis>(func);

  LLVM_DEBUG(dbgs() << "Begin " << DEBUG_TYPE << "\n");

  bool changed = false;

  // Simplify the Tapir loops. There is no reason to simplify everything. The
  // transformation itself doesn't change the loop too much. Even if the
  // simplification somehow results in new inner loops being created, we
  // don't care because the structure of the loop won't have a material impact
  // on the transformation. It may make the prefetch distance analysis more
  // difficult but that is a separate problem.
  //
  // The main reason to simplify the loops is to ensure that all loops have a
  // preheader. It is probably better to just insert the preheader if that is
  // the only thing we really want, but might as well simplify the whole loop
  // in case that turns out to be better for some cases.
  SmallVector<Loop *, 4> wl;
  for (Loop *loop : loopInfo.getLoopsInPreorder()) {
    if (getTaskIfTapirLoop(loop, &taskInfo)) {
      // Check for profitability here because it may need to take into account
      // parent/child/sibling loops and it doesn't make sense to do that
      // in BlockingPrefetchImpl since the latter should only have to care
      // about the loop being transformed.
      if (not ProfitabilityChecker(*loop, func).run())
        continue;

      wl.push_back(loop);
    }
  }

  // It is ok to just process the loops in preorder.
  for (Loop *loop : wl) {
    // If we were to run these preparatory functions within
    // LoopBlockingPrefetchImpl::run(), we would not be able to tell if the
    // loop was changed because it was blocked or simply because these
    // transformations returned true.
    changed |= simplifyLoop(loop, &domTree, &loopInfo, &scEvol, &assCache,
                            nullptr, false);
    changed |= formLCSSARecursively(*loop, domTree, &loopInfo, &scEvol);

    bool blocked =
        LoopBlockingPrefetchImpl(*loop, func, assCache, domTree, loopInfo,
                                 scEvol, taskInfo, &optRemarks)
            .run();

#ifndef NDEBUG
    if (blocked) {
      domTree.verify();

      // The parent is guaranteed to be present after the transformation.
      // The parent itself will only have a parent if the loop had a parent
      // before the transformation. The check is to make sure that in the
      // process of transforming the loop, we haven't damaged the (now)
      // grandparent.
      Loop *parent = loop->getParentLoop();
      parent->verifyLoop();
      if (Loop *grandparent = parent->getParentLoop())
        grandparent->verifyLoop();
    }
#endif

    changed |= blocked;
  }

  LLVM_DEBUG(dbgs() << "End " << DEBUG_TYPE << "\n");

  if (changed)
    taskInfo.recalculate(func, domTree);

  if (not changed)
    return PreservedAnalyses::all();
  return getLoopPassPreservedAnalyses();
}
