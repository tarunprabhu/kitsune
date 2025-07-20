//===- CloningUtils.cpp - Utilities to aid in cloning entities ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to support cloning code from host to embedded modules.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/CloningUtils.h"
#include "kitsune/Core/ReachableGlobals.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

static void cloneGlobalVariablesInto(const ReachableGlobals &globals, TTID tt,
                                     Module &devM, ValueToValueMapTy &vmap,
                                     unsigned asConst, unsigned asNonConst) {
  for (GlobalValue *v : globals) {
    auto *g = dyn_cast<GlobalVariable>(v);
    if (not g)
      continue;

    assert(g->getType()->getAddressSpace() == 0 &&
           "Global variables must be in default address space");

    // It would be nice to do this as a "post-processing" pass, for instance,
    // when preparing the kernel module for PTX. However, we need to pass the
    // name of the global as a string to several Kitsune intrinsics. That would
    // make the job of renaming them later much more complicated since we would
    // have to modify all the calls in the host that use the global variable's
    // name.
    StringRef name = g->getName();
    bool isConst = g->isConstant();
    Type *type = g->getValueType();
    MaybeAlign align = g->getAlign();
    GlobalValue::ThreadLocalMode threadLocalMode = g->getThreadLocalMode();

    GlobalVariable *newg = nullptr;
    if ((newg = devM.getGlobalVariable(name, /*AllowLocal=*/true))) {
      // If a global with the name is already present in the kernel module,
      // another outlined loop in the host module used the same global. The
      // global is already present, so we just need to update VMap correctly.
      // This is done after this if-else block.
    } else if (isConst) {
      // If the global variable is a constant we can clone it into the device
      // module along with its initializer where it will be treated as an
      // internal variable. There is no coordination with the host.
      Constant *init = g->getInitializer();
      newg = new GlobalVariable(
          devM, type, isConst, GlobalValue::InternalLinkage, init, name,
          /*InsertBefore=*/nullptr, threadLocalMode, asConst);
      newg->setDSOLocal(true);
      newg->setAlignment(align);
    } else {
      // If the global is not constant, we will need to create a device-side
      // version that will have the host-side value copied over prior to
      // launching the kernel.
      Constant *init = Constant::getNullValue(type);
      newg = new GlobalVariable(
          devM, type, isConst, GlobalValue::ExternalLinkage, init, name,
          /*InsertBefore=*/nullptr, threadLocalMode, asNonConst);
      newg->setDSOLocal(true);
      newg->setAlignment(align);
    }

    // This is really just a sanity check in case the code above changes and
    // someone makes a silly mistake.
    assert(newg && "All global variables must have a corresponding global "
                   "in the kernel module");

    // The global variables are assumed to be in the default address space when
    // outlining. All uses of the global expect them to be in the default
    // address space. If they are not, cast them in the vmap so when we clone
    // any entities that use them, we do not have type mismatches.
    if (newg->getType()->getAddressSpace()) {
      LLVMContext &ctx = devM.getContext();
      PointerType *ptrTy = PointerType::getUnqual(ctx);
      vmap[g] = ConstantExpr::getAddrSpaceCast(newg, ptrTy);
    } else {
      vmap[g] = newg;
    }
  }
}

static void cloneFunctionsInto(const ReachableGlobals &globals, TTID tt,
                               Module &devM, ValueToValueMapTy &vmap) {
  // Functions that are called from the tapir loop must be cloned into the
  // kernel module, especially if they contain a body. This is a two-step
  // process - first we create a declaration for the functions since these may
  // be called by the other reachable functions. The VMap already contains
  // mappings for the global variables that may be needed.
  for (GlobalValue *g : globals) {
    if (auto *f = dyn_cast<Function>(g)) {
      StringRef fname = f->getName();
      if (not devM.getFunction(fname)) {
        FunctionType *fty = f->getFunctionType();
        GlobalValue::LinkageTypes linkage = f->getLinkage();
        Function *devf = Function::Create(fty, linkage, fname, devM);
        for (unsigned i = 0; i < f->arg_size(); ++i) {
          Argument *a = f->getArg(i);
          Argument *deva = devf->getArg(i);
          deva->setName(a->getName());
          vmap[a] = deva;
        }
      }
      vmap[f] = devM.getFunction(fname);
    }
  }

  // The vmap now contains mappings from all functions in the source module to
  // their counterparts in the device module. It is now safe to clone the bodies
  // of the functions.
  for (GlobalValue *g : globals) {
    if (auto *f = dyn_cast<Function>(g)) {
      if (f->size() and not f->isIntrinsic()) {
        SmallVector<ReturnInst *, 8> returns;
        auto *devf = cast<Function>(vmap[f]);
        CloneFunctionInto(devf, f, vmap,
                          CloneFunctionChangeType::DifferentModule, returns);
        devf->addFnAttr(Attribute::KitDevice);
      }
    }
  }
}

static void cloneIFuncsInto(const ReachableGlobals &globals, TTID tt,
                            Module &devM, ValueToValueMapTy &vmap) {
  // IFunc's are a GNU extension, and it is unlikely that we will ever compile
  // code that uses them.
  for (GlobalValue *v : globals)
    if (isa<GlobalIFunc>(v))
      llvm_unreachable("cloneReachableIFuncsInto: not yet implemented");
}

static void cloneGlobalAliasesInto(const ReachableGlobals &globals, TTID tt,
                                   Module &devM, ValueToValueMapTy &vmap) {
  // FIXME: At some point, we should support global aliases, but right now,
  // there are a number of other features that need to be supported.
  for (GlobalValue *v : globals)
    if (isa<GlobalAlias>(v))
      llvm_unreachable("cloneUsedGlobalAliasesInto: not yet implemented");
}

void llvm::cloneGlobalValuesInto(const ReachableGlobals &globals, TTID tt,
                                 Module &devM, ValueToValueMapTy &vmap) {
  // NVPTX has a number of different address spaces. We do not use them and the
  // code seems to work. It is not clear if there is any advantage to using
  // them, but it may be a good idea to look into it at some point.
  //
  // AMDGPU on the other hand requires the globals to be in specific address
  // spaces
  if (tt == TTID::Hip)
    cloneGlobalVariablesInto(
        globals, tt, devM, vmap,
        /* address space for constant globals */ AMDGPUAS::CONSTANT_ADDRESS,
        /* address space for non-const globals */ AMDGPUAS::GLOBAL_ADDRESS);
  else
    cloneGlobalVariablesInto(globals, tt, devM, vmap, 0, 0);

  // The global variables have to be cloned before cloning the functions because
  // they may be used in the bodies of functions to be cloned.
  cloneFunctionsInto(globals, tt, devM, vmap);
  cloneIFuncsInto(globals, tt, devM, vmap);

  // The aliasee in global aliases is a global value, so they must be cloned
  // after the global variables and functions are in the vmap.
  cloneGlobalAliasesInto(globals, tt, devM, vmap);
}

void llvm::cloneGlobalValuesInto(const ReachableGlobals &globals, TTID tt,
                                 Module &devM) {
  ValueToValueMapTy vmap;
  cloneGlobalValuesInto(globals, tt, devM, vmap);
}
