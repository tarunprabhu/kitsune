//===- GPUTTCommon.cpp - Base class GPU-centric tapir targets -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for the 'cuda' and 'hip' tapir targets.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Targets/GPUTTCommon.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Core/TargetUtils.h"
#include "kitsune/Support/OstreamUtils.h"
#include "kitsune/Support/ToString.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

using namespace llvm;

/// Construct the name for a device module.
static std::string getNameForDeviceModule(TTID tt, const Module &hostM) {
  std::string buf;
  raw_string_ostream os(buf);

  os << "__kit" << tt << "_" << sys::path::filename(hostM.getName());
  os.flush();

  return buf;
}

GPUTTBase::GPUTTBase(TTID tt, Module &hostM, const TTOptions &tto)
    : TapirTarget(hostM, tto), tt(tt), hostM(hostM),
      devM("", hostM.getContext()), nextKernelID(0) {
  assert(isGPUTT(tt) &&
         "Only GPU-centric tapir targets may inherit from GPUTTBase");

  TargetMachine *tm = createTargetMachine(tt, tto);
  devM.setTargetTriple(tm->getTargetTriple());
  devM.setDataLayout(tm->createDataLayout());

  std::string name = getNameForDeviceModule(tt, hostM);
  devM.setModuleIdentifier(name);
  addDeviceModuleFlagsAttr(devM, tt);
  cloneModuleFlagsMetadataInto(devM, hostM);
  cloneIdentMetadataInto(devM, hostM);
}

std::string GPUTTBase::getNameForTapirLoop(const TapirLoopInfo &tl) {
  std::string buf;
  raw_string_ostream os(buf);
  const Loop *loop = tl.getLoop();
  const Function *f = getFunction(*loop);
  const Module *m = f->getParent();

  os << "__kit" << tt << "_loop_";
  if (m->getNamedMetadata("llvm.dbg.cu") || m->getNamedMetadata("llvm.dbg")) {
    // If we have debug info in the module use the line number to name the
    // kernel. This is only to make debugging a shade easier since it makes it
    // easier to associate the kernel function with a loop in source code.
    //
    // FIXME: This is risky. In principle, in a large project, we could have
    // multiple files with the same name in different directories. There is a
    // small possibility that a forall loop occurs on exactly the same line in
    // both of these files. Ideally, we should include the full file path which
    // is guaranteed to be unique. However, that would detract from the
    // "usefulness" of this name (mainly for debugging). For now, we'll stick
    // with this until we can make some of the support tooling more robust to
    // allow us to mangle the name to avoid collisions.
    //
    // There is another issue here where inlining through multiple levels may
    // result in incompatibilities. All this is being done because it makes
    // "IR-dump debugging" easier. This is less of an issue now that parts of
    // the compiler are a lot more stable.
    //
    // TODO: We should consider using a more robust name mangling method to
    // generate function names, or just use the method where loops are just
    // named with a monotonically increasing integer suffix.
    //
    DebugLoc dbgLoc = loop->getStartLoc();
    const DILocation *loc = dbgLoc.get();
    if (const DILocation *inlinedLoc = dbgLoc.getInlinedAt())
      loc = inlinedLoc;
    unsigned line = loc->getLine();
    unsigned col = loc->getColumn();
    StringRef filePath = loc->getFile()->getFilename();
    StringRef fileName = sys::path::filename(filePath);
    os << fileName << "_" << line << "_" << col;
  } else {
    StringRef name = f->getName();
    std::string demangledName;
    if (nonMicrosoftDemangle(name, demangledName,
                             /*CanHaveLeadingDot=*/false,
                             /*ParseParams=*/false))
      os << demangledName;
    else
      os << name;
    os << "_" << nextKernelID;
    ++nextKernelID;
  }

  return buf;
}

Value *GPUTTBase::lowerGrainsizeCall(CallInst *call) {
  // This is only called by the tapir-to-target pass, not by loop-spawning.
  // The tapir-to-target pass should never have anything that will run on a
  // GPU, so fail catastrophically if that ever happens.
  llvm_unreachable(
      "GPUTTBase::lowerGrainsizeCall: did not expect this to be called");
}

void GPUTTBase::lowerSync(SyncInst &si) {
  // This will only be called by the TapirToTarget pass, not by loop spawning,
  // so there is not much that we can do here. A "reasonable" implementation
  // could well be to call Kitsune's sync_stream intrinsic here in case loop
  // spawning is ever modified to actually call this. That way, we can also
  // avoid the sync that is unconditionally added in the processOutlinedLoopCall
  // callback.
  //
  // Until then, fail catastrophically so we know if something unexpectedly
  // changes upstream.
  llvm_unreachable("GPUTTBase::lowerSync: did not expect this to be called");
}

void GPUTTBase::preProcessModule() {
  // Create the global variable that will eventually contain the fat binary of
  // GPU code. This is currently uninitialized, but will be passed to several
  // of the kitsune runtime intrinsic calls when launching kernels, copying
  // global variables from host to device etc.
  (void)createEmbFBGlobal(tt, hostM);
}

void GPUTTBase::postProcessModule() {
  // At this point, we are done with the minimum task of outlining the tapir
  // loop into a kernel module. There are still a number of transformations that
  // must be carried out on this module before it can be compiled to GPU code,
  // but those will be done by subsequent passes. The module here is in a state
  // where we can perform combined host/device analyses and optimizations.
  (void)createEmbBCGlobal(devM, tt, hostM);
}

bool GPUTTBase::preProcessFunction(Function &f, TaskInfo &ti,
                                   bool processingTapirLoops) {
  return false;
}

void GPUTTBase::addHelperAttributes(Function &helper) {
  // This callback is not invoked from LoopSpawning. Only the TapirToTarget
  // pass invokes this. Therefore, any attributes that need to be added to
  // the outlined function are added by the loop outline processors used
  // by this tapir target.
}

void GPUTTBase::preProcessOutlinedTask(Function &f, Instruction *detachPt,
                                       Instruction *tfCreate, bool isSpawner,
                                       BasicBlock *bb) {}

void GPUTTBase::postProcessOutlinedTask(Function &f, Instruction *detachPtr,
                                        Instruction *tfCreate, bool isSpawner,
                                        BasicBlock *tfEntry) {}

void GPUTTBase::preProcessRootSpawner(Function &f, BasicBlock *tfEntry) {}

void GPUTTBase::postProcessRootSpawner(Function &f, BasicBlock *tfEntry) {}

// Process the invocation of a task for an outlined function.  This routine is
// invoked after processSpawner once for each child subtask.
void GPUTTBase::processSubTaskCall(TaskOutlineInfo &toi, DominatorTree &dt) {}

// Process Function f at the end of the lowering process.
void GPUTTBase::postProcessFunction(Function &f, bool outliningTapirLoops) {}

// Process a generated helper Function f produced via outlining, at the end of
// the lowering process.
void GPUTTBase::postProcessHelper(Function &f) {}
