//===- PassManagerBuilder.cpp - Build Standard Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PassManagerBuilder class, which is used to set up a
// "standard" optimization sequence suitable for languages like C and C++.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm-c/Transforms/PassManagerBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Tapir.h"
#include "llvm/Transforms/Tapir/LoopStripMinePass.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;

static cl::opt<bool> EnableTapirLoopStripmine(
    "enable-tapir-loop-stripmine", cl::init(true), cl::Hidden,
    cl::desc("Enable the Tapir loop-stripmining pass (default = on)"));

static cl::opt<bool> EnableSerializeSmallTasks(
  "enable-serialize-small-tasks", cl::Hidden, cl::init(false),
  cl::desc("Serialize any Tapir tasks found to be unprofitable (default = off)"));

static cl::opt<bool> DisableTapirOpts(
    "disable-tapir-opts", cl::init(false), cl::Hidden,
    cl::desc("Disable Tapir optimizations by outlining Tapir tasks early"));

static cl::opt<bool>
    VerifyTapirLowering("verify-tapir-lowering", cl::init(false), cl::Hidden,
                        cl::desc("Verify IR after Tapir lowering steps"));

PassManagerBuilder::PassManagerBuilder() {
    TapirTarget = TapirTargetID::None;
    OptLevel = 2;
    SizeLevel = 0;
    LibraryInfo = nullptr;
    Inliner = nullptr;
    DisableUnrollLoops = false;
    SLPVectorize = false;
    LoopVectorize = true;
    LoopsInterleaved = true;
    LoopStripmine = true;
    LicmMssaOptCap = SetLicmMssaOptCap;
    LicmMssaNoAccForPromotionCap = SetLicmMssaNoAccForPromotionCap;
    DisableGVNLoadPRE = false;
    ForgetAllSCEVInLoopUnroll = ForgetSCEVInLoopUnroll;
    VerifyInput = false;
    VerifyOutput = false;
    MergeFunctions = false;
    DivergentTarget = false;
    CallGraphProfile = true;
}

PassManagerBuilder::~PassManagerBuilder() {
  delete LibraryInfo;
  delete Inliner;
}

void PassManagerBuilder::addInitialAliasAnalysisPasses(
    legacy::PassManagerBase &PM) const {
  // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
  // BasicAliasAnalysis wins if they disagree. This is intended to help
  // support "obvious" type-punning idioms.
  PM.add(createTypeBasedAAWrapperPass());
  PM.add(createScopedNoAliasAAWrapperPass());
}

void PassManagerBuilder::populateFunctionPassManager(
    legacy::FunctionPassManager &FPM) {
  // Add LibraryInfo if we have some.
  if (LibraryInfo)
    FPM.add(new TargetLibraryInfoWrapperPass(*LibraryInfo));

  if (OptLevel == 0) return;

  addInitialAliasAnalysisPasses(FPM);

  // Lower llvm.expect to metadata before attempting transforms.
  // Compare/branch metadata may alter the behavior of passes like SimplifyCFG.
  FPM.add(createLowerExpectIntrinsicPass());
  FPM.add(createCFGSimplificationPass());
  FPM.add(createSROAPass());
  FPM.add(createEarlyCSEPass());
}

void PassManagerBuilder::addFunctionSimplificationPasses(
    legacy::PassManagerBase &MPM) {
  // Start of function pass.
  // Break up aggregate allocas, using SSAUpdater.
  assert(OptLevel >= 1 && "Calling function optimizer with no optimization level!");
  MPM.add(createSROAPass());
  MPM.add(createEarlyCSEPass(true /* Enable mem-ssa. */)); // Catch trivial redundancies

  if (OptLevel > 1) {
    // Speculative execution if the target has divergent branches; otherwise nop.
    MPM.add(createSpeculativeExecutionIfHasBranchDivergencePass());

    MPM.add(createJumpThreadingPass());         // Thread jumps.
    MPM.add(createCorrelatedValuePropagationPass()); // Propagate conditionals
  }
  MPM.add(
      createCFGSimplificationPass(SimplifyCFGOptions().convertSwitchRangeToICmp(
          true))); // Merge & remove BBs
  // Combine silly seq's
  MPM.add(createInstructionCombiningPass());
  if (SizeLevel == 0)
    MPM.add(createLibCallsShrinkWrapPass());

  // TODO: Investigate the cost/benefit of tail call elimination on debugging.
  if (OptLevel > 1)
    MPM.add(createTailCallEliminationPass()); // Eliminate tail calls
  MPM.add(
      createCFGSimplificationPass(SimplifyCFGOptions().convertSwitchRangeToICmp(
          true)));                            // Merge & remove BBs
  MPM.add(createReassociatePass());           // Reassociate expressions

  // Begin the loop pass pipeline.

  // The simple loop unswitch pass relies on separate cleanup passes. Schedule
  // them first so when we re-process a loop they run before other loop
  // passes.
  MPM.add(createLoopInstSimplifyPass());
  MPM.add(createLoopSimplifyCFGPass());

  // Try to remove as much code from the loop header as possible,
  // to reduce amount of IR that will have to be duplicated. However,
  // do not perform speculative hoisting the first time as LICM
  // will destroy metadata that may not need to be destroyed if run
  // after loop rotation.
  // TODO: Investigate promotion cap for O1.
  MPM.add(createLICMPass(LicmMssaOptCap, LicmMssaNoAccForPromotionCap,
                         /*AllowSpeculation=*/false));
  // Rotate Loop - disable header duplication at -Oz
  MPM.add(createLoopRotatePass(SizeLevel == 2 ? 0 : -1, PrepareForLTO));
  if (EnableSerializeSmallTasks)
    MPM.add(createSerializeSmallTasksPass());
  // TODO: Investigate promotion cap for O1.
  MPM.add(createLICMPass(LicmMssaOptCap, LicmMssaNoAccForPromotionCap,
                         /*AllowSpeculation=*/true));
  MPM.add(createSimpleLoopUnswitchLegacyPass(OptLevel == 3));
  // FIXME: We break the loop pass pipeline here in order to do full
  // simplifycfg. Eventually loop-simplifycfg should be enhanced to replace the
  // need for this.
  MPM.add(createCFGSimplificationPass(
      SimplifyCFGOptions().convertSwitchRangeToICmp(true)));
  MPM.add(createInstructionCombiningPass());
  // We resume loop passes creating a second loop pipeline here.
  MPM.add(createLoopIdiomPass());             // Recognize idioms like memset.
  MPM.add(createIndVarSimplifyPass());        // Canonicalize indvars
  MPM.add(createLoopDeletionPass());          // Delete dead loops

  // Unroll small loops and perform peeling.
  MPM.add(createSimpleLoopUnrollPass(OptLevel, DisableUnrollLoops,
                                     ForgetAllSCEVInLoopUnroll));
  // This ends the loop pass pipelines.

  // Break up allocas that may now be splittable after loop unrolling.
  MPM.add(createSROAPass());

  if (OptLevel > 1) {
    MPM.add(createMergedLoadStoreMotionPass()); // Merge ld/st in diamonds
    MPM.add(createGVNPass(DisableGVNLoadPRE));  // Remove redundancies
  }
  MPM.add(createSCCPPass());                  // Constant prop with SCCP

  // Delete dead bit computations (instcombine runs after to fold away the dead
  // computations, and then ADCE will run later to exploit any new DCE
  // opportunities that creates).
  MPM.add(createBitTrackingDCEPass());        // Delete dead bit computations

  // Run instcombine after redundancy elimination to exploit opportunities
  // opened up by them.
  MPM.add(createInstructionCombiningPass());
  if (OptLevel > 1) {
    MPM.add(createJumpThreadingPass());         // Thread jumps
    MPM.add(createCorrelatedValuePropagationPass());
  }
  MPM.add(createAggressiveDCEPass()); // Delete dead instructions

  MPM.add(createMemCpyOptPass());               // Remove memcpy / form memset
  // TODO: Investigate if this is too expensive at O1.
  if (OptLevel > 1) {
    MPM.add(createDeadStoreEliminationPass());  // Delete dead stores
    MPM.add(createLICMPass(LicmMssaOptCap, LicmMssaNoAccForPromotionCap,
                           /*AllowSpeculation=*/true));
  }

  // Merge & remove BBs and sink & hoist common instructions.
  MPM.add(createCFGSimplificationPass(
      SimplifyCFGOptions().hoistCommonInsts(true).sinkCommonInsts(true)));
  // Clean up after everything.
  MPM.add(createInstructionCombiningPass());
}

/// FIXME: Should LTO cause any differences to this set of passes?
void PassManagerBuilder::addVectorPasses(legacy::PassManagerBase &PM,
                                         bool IsFullLTO) {
  PM.add(createLoopVectorizePass(!LoopsInterleaved, !LoopVectorize));

  if (IsFullLTO) {
    // The vectorizer may have significantly shortened a loop body; unroll
    // again. Unroll small loops to hide loop backedge latency and saturate any
    // parallel execution resources of an out-of-order processor. We also then
    // need to clean up redundancies and loop invariant code.
    // FIXME: It would be really good to use a loop-integrated instruction
    // combiner for cleanup here so that the unrolling and LICM can be pipelined
    // across the loop nests.
    PM.add(createLoopUnrollPass(OptLevel, DisableUnrollLoops,
                                ForgetAllSCEVInLoopUnroll));
    PM.add(createWarnMissedTransformationsPass());
  }

  if (EnableSerializeSmallTasks)
    PM.add(createSerializeSmallTasksPass());

  if (!IsFullLTO) {
    // Eliminate loads by forwarding stores from the previous iteration to loads
    // of the current iteration.
    PM.add(createLoopLoadEliminationPass());
  }
  // Cleanup after the loop optimization passes.
  PM.add(createInstructionCombiningPass());

  // Now that we've formed fast to execute loop structures, we do further
  // optimizations. These are run afterward as they might block doing complex
  // analyses and transforms such as what are needed for loop vectorization.

  // Cleanup after loop vectorization, etc. Simplification passes like CVP and
  // GVN, loop transforms, and others have already run, so it's now better to
  // convert to more optimized IR using more aggressive simplify CFG options.
  // The extra sinking transform can create larger basic blocks, so do this
  // before SLP vectorization.
  PM.add(createCFGSimplificationPass(SimplifyCFGOptions()
                                         .forwardSwitchCondToPhi(true)
                                         .convertSwitchRangeToICmp(true)
                                         .convertSwitchToLookupTable(true)
                                         .needCanonicalLoops(false)
                                         .hoistCommonInsts(true)
                                         .sinkCommonInsts(true)));

  if (IsFullLTO) {
    PM.add(createSCCPPass());                 // Propagate exposed constants
    PM.add(createInstructionCombiningPass()); // Clean up again
    PM.add(createBitTrackingDCEPass());
  }

  // Optimize parallel scalar instruction chains into SIMD instructions.
  if (SLPVectorize) {
    PM.add(createSLPVectorizerPass());
  }

  // Enhance/cleanup vector code.
  PM.add(createVectorCombinePass());

  if (!IsFullLTO) {
    PM.add(createInstructionCombiningPass());

    // Unroll small loops
    PM.add(createLoopUnrollPass(OptLevel, DisableUnrollLoops,
                                ForgetAllSCEVInLoopUnroll));

    if (!DisableUnrollLoops) {
      // LoopUnroll may generate some redundency to cleanup.
      PM.add(createInstructionCombiningPass());

      // Runtime unrolling will introduce runtime check in loop prologue. If the
      // unrolled loop is a inner loop, then the prologue will be inside the
      // outer loop. LICM pass can help to promote the runtime check out if the
      // checked value is loop invariant.
      PM.add(createLICMPass(LicmMssaOptCap, LicmMssaNoAccForPromotionCap,
                            /*AllowSpeculation=*/true));
    }

    PM.add(createWarnMissedTransformationsPass());
  }

  // After vectorization and unrolling, assume intrinsics may tell us more
  // about pointer alignments.
  PM.add(createAlignmentFromAssumptionsPass());

  if (IsFullLTO)
    PM.add(createInstructionCombiningPass());
}

void PassManagerBuilder::populateModulePassManager(
    legacy::PassManagerBase &MPM) {
  MPM.add(createAnnotation2MetadataLegacyPass());

  // Allow forcing function attributes as a debugging and tuning aid.
  MPM.add(createForceFunctionAttrsLegacyPass());

  // If all optimizations are disabled, just run the always-inline pass and,
  // if enabled, the function merging pass.
  if (OptLevel == 0) {
    if (Inliner) {
      MPM.add(Inliner);
      Inliner = nullptr;
    }

    // // Add passes to run just before Tapir lowering.
    // addExtensionsToPM(EP_TapirLate, MPM);
    // addExtensionsToPM(EP_TapirLoopEnd, MPM);

    if (TapirTargetID::None != TapirTarget) {
      MPM.add(createTaskCanonicalizePass());
      MPM.add(createLowerTapirToTargetPass());
      // The lowering pass may leave cruft around.  Clean it up.
      MPM.add(createCFGSimplificationPass());
      MPM.add(createAlwaysInlinerLegacyPass());
    }

    // FIXME: The BarrierNoopPass is a HACK! The inliner pass above implicitly
    // creates a CGSCC pass manager, but we don't want to add extensions into
    // that pass manager. To prevent this we insert a no-op module pass to reset
    // the pass manager to get the same behavior as EP_OptimizerLast in non-O0
    // builds. The function merging pass is
    if (MergeFunctions)
      MPM.add(createMergeFunctionsPass());
    return;
  }

  // Add LibraryInfo if we have some.
  if (LibraryInfo)
    MPM.add(new TargetLibraryInfoWrapperPass(*LibraryInfo));

  addInitialAliasAnalysisPasses(MPM);

  bool RerunAfterTapirLowering = false;
  bool TapirHasBeenLowered = (TapirTargetID::None == TapirTarget);

  if (DisableTapirOpts && (TapirTargetID::None != TapirTarget)) {
    MPM.add(createTaskCanonicalizePass());
    MPM.add(createLowerTapirToTargetPass());
    TapirHasBeenLowered = true;
  }

  do {
    RerunAfterTapirLowering =
       !TapirHasBeenLowered && (ParallelLevel > 0) && !PrepareForThinLTO;

  // Infer attributes about declarations if possible.
  MPM.add(createInferFunctionAttrsLegacyPass());

  if (OptLevel > 2)
    MPM.add(createCallSiteSplittingPass());

  MPM.add(createIPSCCPPass());          // IP SCCP
  MPM.add(createCalledValuePropagationPass());

  MPM.add(createGlobalOptimizerPass()); // Optimize out global vars
  // Promote any localized global vars.
  MPM.add(createPromoteMemoryToRegisterPass());

  MPM.add(createDeadArgEliminationPass()); // Dead argument elimination

  MPM.add(createInstructionCombiningPass()); // Clean up after IPCP & DAE
  MPM.add(
      createCFGSimplificationPass(SimplifyCFGOptions().convertSwitchRangeToICmp(
          true))); // Clean up after IPCP & DAE

  // We add a module alias analysis pass here. In part due to bugs in the
  // analysis infrastructure this "works" in that the analysis stays alive
  // for the entire SCC pass run below.
  MPM.add(createGlobalsAAWrapperPass());

  // Start of CallGraph SCC passes.
  bool RunInliner = false;
  if (Inliner) {
    MPM.add(Inliner);
    Inliner = nullptr;
    RunInliner = true;
  }

  MPM.add(createPostOrderFunctionAttrsLegacyPass());

  addFunctionSimplificationPasses(MPM);

  // FIXME: This is a HACK! The inliner pass above implicitly creates a CGSCC
  // pass manager that we are specifically trying to avoid. To prevent this
  // we must insert a no-op module pass to reset the pass manager.
  MPM.add(createBarrierNoopPass());

  if (OptLevel > 1)
    // Remove avail extern fns and globals definitions if we aren't
    // compiling an object file for later LTO. For LTO we want to preserve
    // these so they are eligible for inlining at link-time. Note if they
    // are unreferenced they will be removed by GlobalDCE later, so
    // this only impacts referenced available externally globals.
    // Eventually they will be suppressed during codegen, but eliminating
    // here enables more opportunity for GlobalDCE as it may make
    // globals referenced by available external functions dead
    // and saves running remaining passes on the eliminated functions.
    MPM.add(createEliminateAvailableExternallyPass());

  MPM.add(createReversePostOrderFunctionAttrsPass());

  // The inliner performs some kind of dead code elimination as it goes,
  // but there are cases that are not really caught by it. We might
  // at some point consider teaching the inliner about them, but it
  // is OK for now to run GlobalOpt + GlobalDCE in tandem as their
  // benefits generally outweight the cost, making the whole pipeline
  // faster.
  if (RunInliner) {
    MPM.add(createGlobalOptimizerPass());
    MPM.add(createGlobalDCEPass());
  }

  if (EnableSerializeSmallTasks)
    MPM.add(createSerializeSmallTasksPass());

  // If we are planning to perform ThinLTO later, let's not bloat the code with
  // unrolling/vectorization/... now. We'll first run the inliner + CGSCC passes
  // during ThinLTO and perform the rest of the optimizations afterward.
  if (PrepareForThinLTO) {
    // Ensure we perform any last passes, but do so before renaming anonymous
    // globals in case the passes add any.
    addExtensionsToPM(EP_OptimizerLast, MPM);
    MPM.add(createCanonicalizeAliasesPass());
    // Rename anon globals to be able to export them in the summary.
    MPM.add(createNameAnonGlobalPass());
    return;
  }

  if (PerformThinLTO)
    // Optimize globals now when performing ThinLTO, this enables more
    // optimizations later.
    MPM.add(createGlobalOptimizerPass());

  // Scheduling LoopVersioningLICM when inlining is over, because after that
  // we may see more accurate aliasing. Reason to run this late is that too
  // early versioning may prevent further inlining due to increase of code
  // size. By placing it just after inlining other optimizations which runs
  // later might get benefit of no-alias assumption in clone loop.
  if (UseLoopVersioningLICM) {
    MPM.add(createLoopVersioningLICMPass());    // Do LoopVersioningLICM
    MPM.add(createLICMPass(LicmMssaOptCap, LicmMssaNoAccForPromotionCap,
                           /*AllowSpeculation=*/true));
  }

  // We add a fresh GlobalsModRef run at this point. This is particularly
  // useful as the above will have inlined, DCE'ed, and function-attr
  // propagated everything. We should at this point have a reasonably minimal
  // and richly annotated call graph. By computing aliasing and mod/ref
  // information for all local globals here, the late loop passes and notably
  // the vectorizer will be able to use them to help recognize vectorizable
  // memory operations.
  //
  // Note that this relies on a bug in the pass manager which preserves
  // a module analysis into a function pass pipeline (and throughout it) so
  // long as the first function pass doesn't invalidate the module analysis.
  // Thus both Float2Int and LoopRotate have to preserve AliasAnalysis for
  // this to work. Fortunately, it is trivial to preserve AliasAnalysis
  // (doing nothing preserves it as it is required to be conservatively
  // correct in the face of IR changes).
  MPM.add(createGlobalsAAWrapperPass());

  MPM.add(createFloat2IntPass());
  MPM.add(createLowerConstantIntrinsicsPass());

  if (EnableMatrix) {
    MPM.add(createLowerMatrixIntrinsicsPass());
    // CSE the pointer arithmetic of the column vectors.  This allows alias
    // analysis to establish no-aliasing between loads and stores of different
    // columns of the same matrix.
    MPM.add(createEarlyCSEPass(false));
  }

  // Stripmine Tapir loops.
  if (LoopStripmine) {
    MPM.add(createLoopStripMinePass());
    // Cleanup the IR after stripminning.
    MPM.add(createTaskSimplifyPass());
    MPM.add(createLoopSimplifyCFGPass());
    MPM.add(createIndVarSimplifyPass());        // Canonicalize indvars
    MPM.add(createEarlyCSEPass());
    MPM.add(createJumpThreadingPass());         // Thread jumps
    MPM.add(createCorrelatedValuePropagationPass());
    addInstructionCombiningPass(MPM);
  }

  // addExtensionsToPM(EP_VectorizerStart, MPM);

  // Re-rotate loops in all our loop nests. These may have fallout out of
  // rotated form due to GVN or other transformations, and the vectorizer relies
  // on the rotated form. Disable header duplication at -Oz.
  MPM.add(createLoopRotatePass(SizeLevel == 2 ? 0 : -1, false));

  // Distribute loops to allow partial vectorization.  I.e. isolate dependences
  // into separate loop that would otherwise inhibit vectorization.  This is
  // currently only performed for loops marked with the metadata
  // llvm.loop.distribute=true or when -enable-loop-distribute is specified.
  MPM.add(createLoopDistributePass());

  addVectorPasses(MPM, /* IsFullLTO */ false);

  // FIXME: We shouldn't bother with this anymore.
  MPM.add(createStripDeadPrototypesPass()); // Get rid of dead prototypes

  // GlobalOpt already deletes dead functions and globals, at -O2 try a
  // late pass of GlobalDCE.  It is capable of deleting dead cycles.
  if (OptLevel > 1) {
    MPM.add(createGlobalDCEPass());         // Remove dead fns and globals.
    MPM.add(createConstantMergePass());     // Merge dup global constants
  }

  if (MergeFunctions)
    MPM.add(createMergeFunctionsPass());

  // LoopSink pass sinks instructions hoisted by LICM, which serves as a
  // canonicalization pass that enables other optimizations. As a result,
  // LoopSink pass needs to be a very late IR pass to avoid undoing LICM
  // result too early.
  MPM.add(createLoopSinkPass());
  // Get rid of LCSSA nodes.
  MPM.add(createInstSimplifyLegacyPass());

  // This hoists/decomposes div/rem ops. It should run after other sink/hoist
  // passes to avoid re-sinking, but before SimplifyCFG because it can allow
  // flattening of blocks.
  MPM.add(createDivRemPairsPass());

  // LoopSink (and other loop passes since the last simplifyCFG) might have
  // resulted in single-entry-single-exit or empty blocks. Clean up the CFG.
  MPM.add(createCFGSimplificationPass(
      SimplifyCFGOptions().convertSwitchRangeToICmp(true)));
  MPM.add(createTaskSimplifyPass());

  if (RerunAfterTapirLowering || (TapirTargetID::None == TapirTarget))
    // Add passes to run just before Tapir lowering.
    addExtensionsToPM(EP_TapirLate, MPM);

  if (!TapirHasBeenLowered) {
    // First handle Tapir loops.  First, simplify their induction variables.
    MPM.add(createIndVarSimplifyPass());
    // Re-rotate loops in all our loop nests. These may have fallout out of
    // rotated form due to GVN or other transformations, and loop spawning
    // relies on the rotated form.  Disable header duplication at -Oz.
    MPM.add(createLoopRotatePass(SizeLevel == 2 ? 0 : -1));
    // Outline Tapir loops as needed.
    MPM.add(createLoopSpawningTIPass());
    if (VerifyTapirLowering)
      // Verify the IR produced by loop spawning
      MPM.add(createVerifierPass());

    // The LoopSpawning pass may leave cruft around.  Clean it up.
    MPM.add(createCFGSimplificationPass(
        SimplifyCFGOptions().convertSwitchRangeToICmp(true)));
    MPM.add(createPostOrderFunctionAttrsLegacyPass());
    if (OptLevel > 2)
      MPM.add(createArgumentPromotionPass()); // Scalarize uninlined fn args
    addFunctionSimplificationPasses(MPM);
    MPM.add(createReversePostOrderFunctionAttrsPass());
    if (MergeFunctions)
      MPM.add(createMergeFunctionsPass());
    MPM.add(createBarrierNoopPass());
    // addFunctionSimplificationPasses(MPM);
    addExtensionsToPM(EP_TapirLoopEnd, MPM);

    // Now lower Tapir to Target runtime calls.
    MPM.add(createTaskCanonicalizePass());
    MPM.add(createLowerTapirToTargetPass());
    if (VerifyTapirLowering)
      // Verify the IR produced by Tapir lowering
      MPM.add(createVerifierPass());
    // The lowering pass introduces new functions and may leave cruft around.
    // Clean it up.
    MPM.add(createCFGSimplificationPass(
        SimplifyCFGOptions().convertSwitchRangeToICmp(true)));
    MPM.add(createPostOrderFunctionAttrsLegacyPass());
    if (OptLevel > 2)
      MPM.add(createArgumentPromotionPass()); // Scalarize uninlined fn args
    addFunctionSimplificationPasses(MPM);
    MPM.add(createReversePostOrderFunctionAttrsPass());

    MPM.add(createIPSCCPPass());          // IP SCCP
    MPM.add(createCalledValuePropagationPass());
    MPM.add(createGlobalOptimizerPass()); // Optimize out global vars
    // Promote any localized global vars.
    MPM.add(createPromoteMemoryToRegisterPass());

    MPM.add(createDeadArgEliminationPass()); // Dead argument elimination

    addInstructionCombiningPass(MPM); // Clean up after IPCP & DAE
    MPM.add(createCFGSimplificationPass()); // Clean up after IPCP & DAE

    if (MergeFunctions)
      MPM.add(createMergeFunctionsPass());
    MPM.add(createBarrierNoopPass());

    // We add a module alias analysis pass here. In part due to bugs in the
    // analysis infrastructure this "works" in that the analysis stays alive
    // for the entire SCC pass run below.
    MPM.add(createGlobalsAAWrapperPass());

    // Start of CallGraph SCC passes.
    MPM.add(createPruneEHPass()); // Remove dead EH info
    MPM.add(createAlwaysInlinerLegacyPass());

    MPM.add(createPostOrderFunctionAttrsLegacyPass());
    if (OptLevel > 2)
      MPM.add(createArgumentPromotionPass()); // Scalarize uninlined fn args

    addFunctionSimplificationPasses(MPM);

    // FIXME: This is a HACK! The inliner pass above implicitly creates a CGSCC
    // pass manager that we are specifically trying to avoid. To prevent this
    // we must insert a no-op module pass to reset the pass manager.
    MPM.add(createBarrierNoopPass());

    if (RunPartialInlining)
      MPM.add(createPartialInliningPass());

    if (OptLevel > 1)
      // Remove avail extern fns and globals definitions if we aren't
      // compiling an object file for later LTO. For LTO we want to preserve
      // these so they are eligible for inlining at link-time. Note if they
      // are unreferenced they will be removed by GlobalDCE later, so
      // this only impacts referenced available externally globals.
      // Eventually they will be suppressed during codegen, but eliminating
      // here enables more opportunity for GlobalDCE as it may make
      // globals referenced by available external functions dead
      // and saves running remaining passes on the eliminated functions.
      MPM.add(createEliminateAvailableExternallyPass());

    MPM.add(createReversePostOrderFunctionAttrsPass());

    // The inliner performs some kind of dead code elimination as it goes,
    // but there are cases that are not really caught by it. We might
    // at some point consider teaching the inliner about them, but it
    // is OK for now to run GlobalOpt + GlobalDCE in tandem as their
    // benefits generally outweight the cost, making the whole pipeline
    // faster.
    if (RunInliner) {
      MPM.add(createGlobalOptimizerPass());
      MPM.add(createGlobalDCEPass());
    }

    TapirHasBeenLowered = true;
    // HACK to disable rerun of the pipeline after Tapir lowering.
    RerunAfterTapirLowering = false;
  }
  } while (RerunAfterTapirLowering);

  // addExtensionsToPM(EP_OptimizerLast, MPM);

  MPM.add(createAnnotationRemarksLegacyPass());
}

// void PassManagerBuilder::populateModulePassManager(legacy::PassManagerBase& MPM) {
//   if (ParallelLevel != 0) {
//     switch (ParallelLevel) {
//       case 1: //fcilkplus
//       case 2: //ftapir
//         prepopulateModulePassManager(MPM);
//         addExtensionsToPM(EP_TapirLate, MPM);
//         break;
//       case 3: //fdetach
//         MPM.add(createLowerTapirToCilkPass(ParallelLevel == 2, InstrumentCilk));
//         prepopulateModulePassManager(MPM);
//         addExtensionsToPM(EP_TapirLate, MPM);
//         break;
//       case 0: llvm_unreachable("invalid");
//     }

//     MPM.add(createBarrierNoopPass());

//     if (OptLevel > 0) {
//       MPM.add(createIndVarSimplifyPass());

//       // Re-rotate loops in all our loop nests. These may have fallout out of
//       // rotated form due to GVN or other transformations, and loop spawning
//       // relies on the rotated form.  Disable header duplication at -Oz.
//       MPM.add(createLoopRotatePass(SizeLevel == 2 ? 0 : -1));

//       MPM.add(createLoopSpawningPass());

//       // The LoopSpawning pass may leave cruft around.  Clean it up.
//       MPM.add(createLoopDeletionPass());
//       MPM.add(createCFGSimplificationPass());
//       addInstructionCombiningPass(MPM);
//       addExtensionsToPM(EP_Peephole, MPM);
//     }

//     // if (ParallelLevel != 3) MPM.add(createInferFunctionAttrsLegacyPass());
//     MPM.add(createInferFunctionAttrsLegacyPass());
//     MPM.add(createUnifyFunctionExitNodesPass());
//     MPM.add(createLowerTapirToCilkPass(ParallelLevel == 2, InstrumentCilk));
//     // The lowering pass may leave cruft around.  Clean it up.
//     MPM.add(createCFGSimplificationPass());
//     // if (ParallelLevel != 3) MPM.add(createInferFunctionAttrsLegacyPass());
//     MPM.add(createInferFunctionAttrsLegacyPass());
//     if (OptLevel != 0) MPM.add(createMergeFunctionsPass());
//     MPM.add(createBarrierNoopPass());
//   }
//   prepopulateModulePassManager(MPM);
//   if (ParallelLevel == 0)
//     addExtensionsToPM(EP_TapirLate, MPM);
//   addExtensionsToPM(EP_OptimizerLast, MPM);
// }

void PassManagerBuilder::addLTOOptimizationPasses(legacy::PassManagerBase &PM) {
  // Load sample profile before running the LTO optimization pipeline.
  if (!PGOSampleUse.empty()) {
    PM.add(createPruneEHPass());
    PM.add(createSampleProfileLoaderPass(PGOSampleUse));
  }

  // Remove unused virtual tables to improve the quality of code generated by
  // whole-program devirtualization and bitset lowering.
  PM.add(createGlobalDCEPass());

  // Provide AliasAnalysis services for optimizations.
  addInitialAliasAnalysisPasses(PM);

  // Allow forcing function attributes as a debugging and tuning aid.
  PM.add(createForceFunctionAttrsLegacyPass());

  // Infer attributes about declarations if possible.
  PM.add(createInferFunctionAttrsLegacyPass());

  if (OptLevel > 1) {
    // Split call-site with more constrained arguments.
    PM.add(createCallSiteSplittingPass());

    // Indirect call promotion. This should promote all the targets that are
    // left by the earlier promotion pass that promotes intra-module targets.
    // This two-step promotion is to save the compile time. For LTO, it should
    // produce the same result as if we only do promotion here.
    PM.add(
        createPGOIndirectCallPromotionLegacyPass(true, !PGOSampleUse.empty()));

    // Propage constant function arguments by specializing the functions.
    if (EnableFunctionSpecialization && OptLevel > 2)
      PM.add(createFunctionSpecializationPass());

    // Propagate constants at call sites into the functions they call.  This
    // opens opportunities for globalopt (and inlining) by substituting function
    // pointers passed as arguments to direct uses of functions.
    PM.add(createIPSCCPPass());

    // Attach metadata to indirect call sites indicating the set of functions
    // they may target at run-time. This should follow IPSCCP.
    PM.add(createCalledValuePropagationPass());

    // Infer attributes on declarations, call sites, arguments, etc.
    if (AttributorRun & AttributorRunOption::MODULE)
      PM.add(createAttributorLegacyPass());
  }

  // Infer attributes about definitions. The readnone attribute in particular is
  // required for virtual constant propagation.
  PM.add(createPostOrderFunctionAttrsLegacyPass());
  PM.add(createReversePostOrderFunctionAttrsPass());

  // Split globals using inrange annotations on GEP indices. This can help
  // improve the quality of generated code when virtual constant propagation or
  // control flow integrity are enabled.
  PM.add(createGlobalSplitPass());

  // Apply whole-program devirtualization and virtual constant propagation.
  PM.add(createWholeProgramDevirtPass(ExportSummary, nullptr));

  // That's all we need at opt level 1.
  if (OptLevel == 1)
    return;

  // Now that we internalized some globals, see if we can hack on them!
  PM.add(createGlobalOptimizerPass());
  // Promote any localized global vars.
  PM.add(createPromoteMemoryToRegisterPass());

  // Linking modules together can lead to duplicated global constants, only
  // keep one copy of each constant.
  PM.add(createConstantMergePass());

  // Remove unused arguments from functions.
  PM.add(createDeadArgEliminationPass());

  // Reduce the code after globalopt and ipsccp.  Both can open up significant
  // simplification opportunities, and both can propagate functions through
  // function pointers.  When this happens, we often have to resolve varargs
  // calls, etc, so let instcombine do this.
  if (OptLevel > 2)
    PM.add(createAggressiveInstCombinerPass());
  PM.add(createInstructionCombiningPass());
  addExtensionsToPM(EP_Peephole, PM);

  // Inline small functions
  bool RunInliner = Inliner;
  if (RunInliner) {
    PM.add(Inliner);
    Inliner = nullptr;
  }

  PM.add(createPruneEHPass());   // Remove dead EH info.

  // CSFDO instrumentation and use pass.
  addPGOInstrPasses(PM, /* IsCS */ true);

  // Infer attributes on declarations, call sites, arguments, etc. for an SCC.
  if (AttributorRun & AttributorRunOption::CGSCC)
    PM.add(createAttributorCGSCCLegacyPass());

  // Try to perform OpenMP specific optimizations. This is a (quick!) no-op if
  // there are no OpenMP runtime calls present in the module.
  if (OptLevel > 1)
    PM.add(createOpenMPOptCGSCCLegacyPass());

  // Optimize globals again if we ran the inliner.
  if (RunInliner)
    PM.add(createGlobalOptimizerPass());
  PM.add(createGlobalDCEPass()); // Remove dead functions.

  // If we didn't decide to inline a function, check to see if we can
  // transform it to pass arguments by value instead of by reference.
  PM.add(createArgumentPromotionPass());

  // The IPO passes may leave cruft around.  Clean up after them.
  PM.add(createInstructionCombiningPass());
  addExtensionsToPM(EP_Peephole, PM);
  PM.add(createJumpThreadingPass(/*FreezeSelectCond*/ true));

  // Break up allocas
  PM.add(createSROAPass());

  // LTO provides additional opportunities for tailcall elimination due to
  // link-time inlining, and visibility of nocapture attribute.
  if (OptLevel > 1)
    PM.add(createTailCallEliminationPass());

  // Infer attributes on declarations, call sites, arguments, etc.
  PM.add(createPostOrderFunctionAttrsLegacyPass()); // Add nocapture.
  // Run a few AA driven optimizations here and now, to cleanup the code.
  PM.add(createGlobalsAAWrapperPass()); // IP alias analysis.

  PM.add(createLICMPass(LicmMssaOptCap, LicmMssaNoAccForPromotionCap,
                        /*AllowSpeculation=*/true));
  PM.add(NewGVN ? createNewGVNPass()
                : createGVNPass(DisableGVNLoadPRE)); // Remove redundancies.
  PM.add(createMemCpyOptPass());            // Remove dead memcpys.

  // Nuke dead stores.
  PM.add(createDeadStoreEliminationPass());
  PM.add(createMergedLoadStoreMotionPass()); // Merge ld/st in diamonds.

  // More loops are countable; try to optimize them.
  if (EnableLoopFlatten)
    PM.add(createLoopFlattenPass());
  PM.add(createIndVarSimplifyPass());
  PM.add(createLoopDeletionPass());
  if (EnableLoopInterchange)
    PM.add(createLoopInterchangePass());

  if (EnableConstraintElimination)
    PM.add(createConstraintEliminationPass());

  // Unroll small loops and perform peeling.
  PM.add(createSimpleLoopUnrollPass(OptLevel, DisableUnrollLoops,
                                    ForgetAllSCEVInLoopUnroll));
  PM.add(createLoopDistributePass());

  addVectorPasses(PM, /* IsFullLTO */ true);

  addExtensionsToPM(EP_Peephole, PM);

  PM.add(createJumpThreadingPass(/*FreezeSelectCond*/ true));
}

void PassManagerBuilder::addLateLTOOptimizationPasses(
    legacy::PassManagerBase &PM) {
  // See comment in the new PM for justification of scheduling splitting at
  // this stage (\ref buildLTODefaultPipeline).
  if (EnableHotColdSplit)
    PM.add(createHotColdSplittingPass());

  // Delete basic blocks, which optimization passes may have killed.
  PM.add(
      createCFGSimplificationPass(SimplifyCFGOptions().hoistCommonInsts(true)));

  // Drop bodies of available externally objects to improve GlobalDCE.
  PM.add(createEliminateAvailableExternallyPass());

  // Now that we have optimized the program, discard unreachable functions.
  PM.add(createGlobalDCEPass());

  // FIXME: this is profitable (for compiler time) to do at -O0 too, but
  // currently it damages debug info.
  if (MergeFunctions)
    PM.add(createMergeFunctionsPass());
}

void PassManagerBuilder::populateThinLTOPassManager(
    legacy::PassManagerBase &PM) {
  PerformThinLTO = true;
  if (LibraryInfo)
    PM.add(new TargetLibraryInfoWrapperPass(*LibraryInfo));

  if (VerifyInput)
    PM.add(createVerifierPass());

  if (ImportSummary) {
    // This pass imports type identifier resolutions for whole-program
    // devirtualization and CFI. It must run early because other passes may
    // disturb the specific instruction patterns that these passes look for,
    // creating dependencies on resolutions that may not appear in the summary.
    //
    // For example, GVN may transform the pattern assume(type.test) appearing in
    // two basic blocks into assume(phi(type.test, type.test)), which would
    // transform a dependency on a WPD resolution into a dependency on a type
    // identifier resolution for CFI.
    //
    // Also, WPD has access to more precise information than ICP and can
    // devirtualize more effectively, so it should operate on the IR first.
    PM.add(createWholeProgramDevirtPass(nullptr, ImportSummary));
    PM.add(createLowerTypeTestsPass(nullptr, ImportSummary));
  }

  populateModulePassManager(PM);

  if (VerifyOutput)
    PM.add(createVerifierPass());
  PerformThinLTO = false;
}

void PassManagerBuilder::populateLTOPassManager(legacy::PassManagerBase &PM) {
  if (LibraryInfo)
    PM.add(new TargetLibraryInfoWrapperPass(*LibraryInfo));

  if (VerifyInput)
    PM.add(createVerifierPass());

  addExtensionsToPM(EP_FullLinkTimeOptimizationEarly, PM);

  if (OptLevel != 0)
    addLTOOptimizationPasses(PM);
  else {
    // The whole-program-devirt pass needs to run at -O0 because only it knows
    // about the llvm.type.checked.load intrinsic: it needs to both lower the
    // intrinsic itself and handle it in the summary.
    PM.add(createWholeProgramDevirtPass(ExportSummary, nullptr));
  }

  // Create a function that performs CFI checks for cross-DSO calls with targets
  // in the current module.
  PM.add(createCrossDSOCFIPass());

  // Lower type metadata and the type.test intrinsic. This pass supports Clang's
  // control flow integrity mechanisms (-fsanitize=cfi*) and needs to run at
  // link time if CFI is enabled. The pass does nothing if CFI is disabled.
  PM.add(createLowerTypeTestsPass(ExportSummary, nullptr));
  // Run a second time to clean up any type tests left behind by WPD for use
  // in ICP (which is performed earlier than this in the regular LTO pipeline).
  PM.add(createLowerTypeTestsPass(nullptr, nullptr, true));

  if (OptLevel != 0)
    addLateLTOOptimizationPasses(PM);

  addExtensionsToPM(EP_FullLinkTimeOptimizationLast, PM);

  PM.add(createAnnotationRemarksLegacyPass());

  if (VerifyOutput)
    PM.add(createVerifierPass());
}

LLVMPassManagerBuilderRef LLVMPassManagerBuilderCreate() {
  PassManagerBuilder *PMB = new PassManagerBuilder();
  return wrap(PMB);
}

void LLVMPassManagerBuilderDispose(LLVMPassManagerBuilderRef PMB) {
  PassManagerBuilder *Builder = unwrap(PMB);
  delete Builder;
}

void
LLVMPassManagerBuilderSetOptLevel(LLVMPassManagerBuilderRef PMB,
                                  unsigned OptLevel) {
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->OptLevel = OptLevel;
}

void
LLVMPassManagerBuilderSetSizeLevel(LLVMPassManagerBuilderRef PMB,
                                   unsigned SizeLevel) {
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->SizeLevel = SizeLevel;
}

void
LLVMPassManagerBuilderSetDisableUnitAtATime(LLVMPassManagerBuilderRef PMB,
                                            LLVMBool Value) {
  // NOTE: The DisableUnitAtATime switch has been removed.
}

void
LLVMPassManagerBuilderSetDisableUnrollLoops(LLVMPassManagerBuilderRef PMB,
                                            LLVMBool Value) {
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->DisableUnrollLoops = Value;
}

void
LLVMPassManagerBuilderSetDisableSimplifyLibCalls(LLVMPassManagerBuilderRef PMB,
                                                 LLVMBool Value) {
  // NOTE: The simplify-libcalls pass has been removed.
}

void
LLVMPassManagerBuilderUseInlinerWithThreshold(LLVMPassManagerBuilderRef PMB,
                                              unsigned Threshold) {
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->Inliner = createFunctionInliningPass(Threshold);
}

void
LLVMPassManagerBuilderPopulateFunctionPassManager(LLVMPassManagerBuilderRef PMB,
                                                  LLVMPassManagerRef PM) {
  PassManagerBuilder *Builder = unwrap(PMB);
  legacy::FunctionPassManager *FPM = unwrap<legacy::FunctionPassManager>(PM);
  Builder->populateFunctionPassManager(*FPM);
}

void
LLVMPassManagerBuilderPopulateModulePassManager(LLVMPassManagerBuilderRef PMB,
                                                LLVMPassManagerRef PM) {
  PassManagerBuilder *Builder = unwrap(PMB);
  legacy::PassManagerBase *MPM = unwrap(PM);
  Builder->populateModulePassManager(*MPM);
}
