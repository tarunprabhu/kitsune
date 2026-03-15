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
#include "GPUTTUtils.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TargetUtils.h"
#include "kitsune/Support/TTIDUtils.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

static StringRef getDeviceModuleNamePrefix(TTID tt) {
  switch (tt) {
  case TTID::Cuda:
    return "__kitnv_";
  case TTID::Hip:
    return "__kitamd_";
  default:
    break;
  }
  llvm_unreachable("getDeviceModuleNamePrefix: TTID not handled");
}

GPUTTBase::GPUTTBase(TTID tt, Module &hostM, const TTOptions &tto)
    : TapirTarget(hostM, tto), tt(tt), hostM(hostM),
      devM("", hostM.getContext()), nextKernelID(0) {
  assert(isGPUTT(tt) &&
         "Only GPU-centric tapir targets may inherit from GPUTTBase");

  TargetMachine *tm = createTargetMachine(tt, tto);
  devM.setTargetTriple(tm->getTargetTriple());
  devM.setDataLayout(tm->createDataLayout());

  StringRef pfx = getDeviceModuleNamePrefix(tt);
  std::string name = getNameForDeviceModule(hostM, pfx);
  devM.setModuleIdentifier(name);
  addDeviceModuleFlagsAttr(devM, tt);
  cloneModuleFlagsMetadataInto(devM, hostM);
  cloneIdentMetadataInto(devM, hostM);
}

Constant *GPUTTBase::getConstGrainsize(Type *ty) {
  return ConstantInt::get(ty, 1, /*isSigned=*/false);
}

Value *GPUTTBase::lowerGrainsizeCall(CallInst *call) {
  Value *gs = getConstGrainsize(call->getType());
  call->replaceAllUsesWith(gs);
  call->eraseFromParent();
  return gs;
}

void GPUTTBase::lowerSync(SyncInst &si) {
  // This will only be called by the TapirToTarget pass, not by loop spawning,
  // so there is not much that we can do here. A "reasonable" implementation
  // could well be to call Kitsune's sync_stream intrinsic here in case loop
  // spawning is ever modified to actually call this. That way, we can also
  // avoid the sync that is unconditionally added in the processOutlinedLoopCall
  // callback.
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
