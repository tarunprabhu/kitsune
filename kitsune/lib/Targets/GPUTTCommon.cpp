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
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Core/TargetUtils.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

using namespace llvm;

std::string llvm::normalizeSymbolName(StringRef name, StringRef prefix,
                                      StringRef suffix) {
  auto isInvalidChar = [](char c) -> bool {
    return !std::isalpha(c) && !std::isdigit(c) && c != '_';
  };

  if (std::none_of(name.begin(), name.end(), isInvalidChar))
    return name.str();

  std::string buf;
  llvm::raw_string_ostream os(buf);

  os << prefix;
  for (char c : name)
    os << (isInvalidChar(c) ? '_' : c);
  os << suffix;
  os.flush();

  return buf;
}

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

static std::string tryDemangle(StringRef fname) {
  std::string demangledName;
  if (nonMicrosoftDemangle(fname, demangledName, /*CanHaveLeadingDot=*/false,
                           /*ParseParams=*/false))
    return demangledName;
  return fname.str();
}

// Use debug information to construct a name of the form
// "<filename>:<line>:<col>".
static std::string getNameFromDebugLoc(DebugLoc dbgLoc) {
  std::string buf;
  raw_string_ostream os(buf);

  const DILocation *loc = dbgLoc.get();
  if (const DILocation *inlinedLoc = dbgLoc.getInlinedAt())
    loc = inlinedLoc;
  unsigned line = loc->getLine();
  unsigned col = loc->getColumn();
  StringRef filePath = loc->getFile()->getFilename();
  StringRef fileName = sys::path::filename(filePath);

  os << fileName << "_" << line << "_" << col;
  os.flush();

  return buf;
}

std::string GPUTTBase::getNameForTapirLoop(const TapirLoopInfo &tl) {
  std::string buf;
  raw_string_ostream os(buf);
  const Loop *loop = tl.getLoop();
  const Function *f = getFunction(*loop);

  os << "__kit" << tt << "_loop_";
  if (std::optional<StringRef> name = getNameAttr(*loop))
    os << *name;
  else if (DebugLoc dbgLoc = loop->getStartLoc())
    os << getNameFromDebugLoc(dbgLoc);
  else
    os << tryDemangle(f->getName());

  // We always append a monotonically increasing integer because inlining may
  // may result in the same loop being duplicated in several places. Each of
  // them will have the same name. Even when the debug information is used to
  // generate the kernel name, we may end up with multiple kernels getting the
  // same name. Since there is only a single instance of this object used for
  // the whole module, and functions are not processed in parallel,
  // `nextKernelID` is guaranteed to be different for each loop.
  //
  // FIXME: This only works for now because we do not yet support separate
  // compilation. When we do, we might have a situation where kernels in
  // different translation units end up with the same name. While this is
  // unlikely, it is not impossible. We probably want to use a slightly more
  // "robust" mechanism - but any use of, say, hash functions, random numbers,
  // or the system clock is still not guaranteed to be unique.
  os << "_" << nextKernelID;
  os.flush();

  ++nextKernelID;

  return normalizeSymbolName(buf);
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
