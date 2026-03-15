//===- GPUTTLoop.h - GPU-centric loop outline processors -------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base classes for loop outline processors used by the 'cuda' and 'hip' tapir
// targets.
//
//===----------------------------------------------------------------------===//

#include "GPUTTLoop.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/KernelProperties.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Core/ValueUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

GPUTTLoopBase::GPUTTLoopBase(Module &hostM, Module &devM, const TTOptions &tto,
                             TTID tt, const TapirLoopInfo &tl,
                             StringRef kernelName)
    : LoopOutlineProcessor(hostM, devM, tto,
                           CloneFunctionChangeType::DifferentModule),
      tt(tt), hostM(hostM), devM(devM), kernelName(kernelName) {
  unsigned depth = getPerfectDepthAttr(*tl.getLoop()).value_or(0);
  assert(depth >= 1 && depth <= 3 &&
         "Perfect depth of tapir loop must be in the range [1,3]");

  this->kernelDepth = depth;
}

void GPUTTLoopBase::populateUsedGlobalValues(GlobalVariable &g) {
  usedGlobalValues.insert(&g);
  if (g.hasInitializer())
    populateUsedGlobalValues(*g.getInitializer());
}

void GPUTTLoopBase::populateUsedGlobalValues(GlobalIFunc &g) {
  usedGlobalValues.insert(&g);
  llvm_unreachable("populateUsedGlobalValues: GNU IFUNC not yet supported");
}

void GPUTTLoopBase::populateUsedGlobalValues(GlobalAlias &g) {
  usedGlobalValues.insert(&g);
  llvm_unreachable("populateUsedGlobalValues: GlobalAlias not yet supported");
}

void GPUTTLoopBase::populateUsedGlobalValues(BlockAddress &blkAddr) {
  if (Function *f = blkAddr.getFunction())
    populateUsedGlobalValues(*f);
  if (BasicBlock *bb = blkAddr.getBasicBlock())
    populateUsedGlobalValues(*bb);
}

void GPUTTLoopBase::populateUsedGlobalValues(Constant &c) {
  if (GlobalValue *g = dyn_cast<GlobalValue>(&c))
    if (usedGlobalValues.find(g) != usedGlobalValues.end())
      return;

  if (auto *f = dyn_cast<Function>(&c))
    return populateUsedGlobalValues(*f);
  else if (auto *g = dyn_cast<GlobalVariable>(&c))
    return populateUsedGlobalValues(*g);
  else if (auto *g = dyn_cast<GlobalAlias>(&c))
    return populateUsedGlobalValues(*g);
  else if (auto *g = dyn_cast<GlobalIFunc>(&c))
    return populateUsedGlobalValues(*g);
  else if (auto *blkAddr = dyn_cast<BlockAddress>(&c))
    return populateUsedGlobalValues(*blkAddr);
  else
    for (Use &op : c.operands())
      if (auto *cop = dyn_cast<Constant>(op))
        populateUsedGlobalValues(*cop);
}

void GPUTTLoopBase::populateUsedGlobalValues(BasicBlock &bb) {
  for (Instruction &inst : bb)
    for (Use &op : inst.operands())
      if (auto *c = dyn_cast<Constant>(&op))
        populateUsedGlobalValues(*c);
}

void GPUTTLoopBase::populateUsedGlobalValues(Function &f) {
  usedGlobalValues.insert(&f);
  for (BasicBlock &bb : f)
    populateUsedGlobalValues(bb);
}

void GPUTTLoopBase::populateUsedGlobalValues(Loop &loop) {
  // Collect the globals used in any subloops.
  for (Loop *subLoop : loop)
    for (BasicBlock *bb : subLoop->blocks())
      populateUsedGlobalValues(*bb);

  // Collect the globals used within the loop itself.
  for (BasicBlock *bb : loop.blocks())
    populateUsedGlobalValues(*bb);
}

void GPUTTLoopBase::copyNonConstGlobals(IRBuilder<> &builder,
                                        Intrinsic::ID copyFn) {
  const DataLayout &dl = hostM.getDataLayout();
  LLVMContext &ctx = hostM.getContext();
  Type *i64Ty = Type::getInt64Ty(ctx);
  Type *voidTy = Type::getVoidTy(ctx);
  PointerType *ptrTy = PointerType::getUnqual(ctx);

  GlobalVariable *fb = getEmbFBGlobal(tt, hostM);
  assert(fb && "Embedded fat binary must exist");

  Constant *ctt = toConstant(tt, ctx);
  for (GlobalValue *gv : usedGlobalValues) {
    if (auto *g = dyn_cast<GlobalVariable>(gv)) {
      if (g->isConstant())
        continue;

      GlobalVariable *name = createConstString(g->getName(), hostM);
      Type *type = g->getValueType();
      size_t size = dl.getTypeAllocSize(type);
      Constant *bytes = ConstantInt::get(i64Ty, size);

      Value *devPtr = builder.CreateIntrinsic(
          ptrTy, Intrinsic::kit_symbol_device_ptr, {ctt, fb, name});
      if (copyFn == Intrinsic::kit_symbol_memcpy_dtoh)
        (void)builder.CreateIntrinsic(voidTy, copyFn, {ctt, g, devPtr, bytes});
      else if (copyFn == Intrinsic::kit_symbol_memcpy_htod)
        (void)builder.CreateIntrinsic(voidTy, copyFn, {ctt, devPtr, g, bytes});
      else
        llvm_unreachable("copyNonConstGlobals: Invalid intrinsic");
    }
  }
}

void GPUTTLoopBase::copyNonConstGlobalsDToH(IRBuilder<> &builder) {
  copyNonConstGlobals(builder, Intrinsic::kit_symbol_memcpy_dtoh);
}

void GPUTTLoopBase::copyNonConstGlobalsHToD(IRBuilder<> &builder) {
  copyNonConstGlobals(builder, Intrinsic::kit_symbol_memcpy_htod);
}

void GPUTTLoopBase::cloneUsedGlobalAliases(ValueToValueMapTy &vmap) {
  // FIXME: At some point, we should support global aliases, but right now,
  // there are a number of other features that need to be supported.
  for (GlobalValue *v : usedGlobalValues)
    if (isa<GlobalAlias>(v))
      llvm_unreachable("cloneUsedGlobalAliasesInto: not yet implemented");
}

void GPUTTLoopBase::cloneReachableFuncs(ValueToValueMapTy &vmap) {
  // Functions that are called from the tapir loop must be cloned into the
  // device module, especially if they contain a body. This is a two-step
  // process - first we create a declaration for the functions since these may
  // be called by the other reachable functions. The vmap already contains
  // mappings for the global variables that may be needed.
  for (GlobalValue *g : usedGlobalValues) {
    if (auto *f = dyn_cast<Function>(g)) {
      StringRef fname = f->getName();
      Function *devf = devM.getFunction(fname);
      if (not devf) {
        FunctionType *fty = f->getFunctionType();
        GlobalValue::LinkageTypes linkage = f->getLinkage();
        devf = Function::Create(fty, linkage, fname, devM);
        for (unsigned i = 0; i < f->arg_size(); ++i) {
          Argument *a = f->getArg(i);
          Argument *deva = devf->getArg(i);
          deva->setName(a->getName());
          vmap[a] = deva;
        }
      }
      vmap[f] = devf;
    }
  }

  // The vmap now contains mappings from all functions in the source module to
  // their counterparts in the device module. It is now safe to clone the bodies
  // of the functions.
  for (GlobalValue *g : usedGlobalValues) {
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

GlobalVariable *GPUTTLoopBase::cloneGlobalVariable(GlobalVariable &g) {
  StringRef name = g.getName();
  bool isConst = g.isConstant();
  Type *type = g.getValueType();
  MaybeAlign align = g.getAlign();
  GlobalValue::ThreadLocalMode threadLocalMode = g.getThreadLocalMode();
  GlobalValue::LinkageTypes linkage = GlobalValue::ExternalLinkage;
  Constant *init = Constant::getNullValue(type);
  unsigned addrSpace = getNonConstAddrSpace();

  if (isConst) {
    linkage = GlobalValue::InternalLinkage;
    init = g.getInitializer();
    addrSpace = getConstAddrSpace();
  }

  GlobalVariable *newg =
      new GlobalVariable(devM, type, isConst, linkage, init, name,
                         /*InsertBefore=*/nullptr, threadLocalMode, addrSpace);
  newg->setDSOLocal(true);
  newg->setAlignment(align);

  return newg;
}

void GPUTTLoopBase::cloneUsedGlobalVariables(ValueToValueMapTy &vmap) {
  for (GlobalValue *v : usedGlobalValues) {
    auto *g = dyn_cast<GlobalVariable>(v);
    if (not g)
      continue;

    assert(g->getType()->getAddressSpace() == 0 &&
           "Global variables must be in default address space");

    // If a global with the name is already present in the kernel module,
    // another outlined loop in the host module used the same global.
    StringRef name = g->getName();
    GlobalVariable *newg = devM.getGlobalVariable(name, /*AllowLocal=*/true);
    if (!newg)
      newg = cloneGlobalVariable(*g);

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

void GPUTTLoopBase::cloneReachableIFuncs(ValueToValueMapTy &vmap) {
  // IFunc's are a GNU extension, and it is unlikely that we will ever compile
  // code that uses them.
  for (GlobalValue *v : usedGlobalValues)
    if (isa<GlobalIFunc>(v))
      llvm_unreachable("cloneReachableIFuncsInto: not yet implemented");
}

GlobalVariable *GPUTTLoopBase::getDevGlobal(GlobalVariable *g,
                                            const ValueToValueMapTy &vmap) {
  return cast<GlobalVariable>(stripCasts(cast<Constant>(vmap.lookup(g))));
}

unsigned GPUTTLoopBase::getOpIndex(const Instruction &inst, Value *v) {
  for (unsigned i = 0; i < inst.getNumOperands(); ++i)
    if (inst.getOperand(i) == v)
      return i;
  llvm_unreachable("Trip count not used in loop condition");
}

Value *GPUTTLoopBase::getGrainsize(Type *ty) {
  return ConstantInt::get(ty, 1, /*isSigned=*/false);
}

void GPUTTLoopBase::setKernelFuncLinkage(Function &f) {
  f.setLinkage(GlobalValue::ExternalLinkage);
}

void GPUTTLoopBase::setupLoopControlArgs(TapirLoopInfo *tl,
                                         SmallVectorImpl<Value *> &lcArgs,
                                         SmallVectorImpl<Value *> &lcInputs) {
  InductionDescriptor ivDescr = tl->getPrimaryInduction().second;

  // It is not clear if we actually need the step value to be 1, but until we
  // can be sure of it, we'll be conservative and require it here.
  assert(ivDescr.getStep()->isOne() &&
         "Step of tapir loop induction variable must be 1");

  // We require tapir loops to be lowered to the GPU to have canonical
  // induction variables. This should have been checked before we get here, but
  // make sure that is the case.
  Value *ivBeg = ivDescr.getStartValue();
  assert(isZero(ivBeg) &&
         "Start value of tapir loop induction variable must be 0");

  Value *tc = tl->getTripCount();
  assert(tc && "No trip count found for Tapir loop end argument.");

  // Since the start value is 0, we don't strictly need this. However, not
  // passing this causes issues in loop spawning since that assumes that this
  // value will be passed. The fixes needed to make this work in loop spawning
  // are not particularly difficult, but it does feel messy. For now, we just
  // pass it since the fix to loop spawning will likely require some more
  // thought.
  LoopCtlArgs.push_back(new Argument(ivBeg->getType(), "iv0.x"));
  lcArgs.push_back(LoopCtlArgs.back());
  lcInputs.push_back(ivBeg);

  LoopCtlArgs.push_back(new Argument(tc->getType(), "tc.x"));
  lcArgs.push_back(LoopCtlArgs.back());
  lcInputs.push_back(tc);
}

void GPUTTLoopBase::preProcessTapirLoop(TapirLoopInfo &tl,
                                        ValueToValueMapTy &vmap) {
  // Collect the top-level entities (Function, GlobalVariable, GlobalAlias
  // and GlobalIFunc) that are used in the outlined loop. Since the outlined
  // loop will live in the device module, any GlobalValue's used in it must be
  // be cloned into the device Module. They may also have to be registered with
  // the GPU's runtime, but that will be done in a later pass.
  populateUsedGlobalValues(*tl.getLoop());

  // The global variables have to be cloned first because they may be used in
  // the bodies of the the functions to be cloned. Global aliases must be cloned
  // last because the aliasees must already be in the vmap before they can be
  // cloned.
  cloneUsedGlobalVariables(vmap);
  cloneReachableFuncs(vmap);
  cloneReachableIFuncs(vmap);
  cloneUsedGlobalAliases(vmap);
}

void GPUTTLoopBase::processOutlinedIVs(Function &f, TapirLoopInfo &tl,
                                       ValueToValueMapTy &vmap) {
  Loop *loop = tl.getLoop();

  BasicBlock *bbEntry = cast<BasicBlock>(vmap[loop->getLoopPreheader()]);
  BasicBlock *bbHeader = cast<BasicBlock>(vmap[loop->getHeader()]);
  BasicBlock *bbExit = cast<BasicBlock>(vmap[tl.getExitBlock()]);
  PHINode *iv = cast<PHINode>(vmap[tl.getPrimaryInduction().first]);
  Type *ivType = iv->getType();

  // The outlined loop runs from [iv0.x, tc.x] where iv0.x and tc.x are bounds
  // provided as arguments to the kernel function. Convert these to use
  // threadIdx, blockIdx, blockDim etc.
  //
  // This is the classic calculation for the induction variable i:
  //
  //     i = blockDim.x * blockIdx.x + threadDix.x
  //
  // The calculation below assumes that iv0.x == 0.This is enforced by the rest
  // of this code and is unlikely to ever change. The runtime will also check
  // that this invariant holds. If we ever get a non-zero value, there is a lot
  // that will, at the very least, have to be rethought.
  IRBuilder<> builder(bbEntry->getTerminator());
  Value *tidX =
      builder.CreateIntrinsic(Intrinsic::kit_gpu_thread_id_x, {}, {}, "tid.x");
  Value *bidX =
      builder.CreateIntrinsic(Intrinsic::kit_gpu_block_id_x, {}, {}, "bid.x");
  Value *bszX =
      builder.CreateIntrinsic(Intrinsic::kit_gpu_block_size_x, {}, {}, "bsz.x");
  Value *bdxbi = builder.CreateMul(bszX, bidX);
  Value *tipbdxbi = builder.CreateAdd(bdxbi, tidX, ".ivbeg.x");

  // This is the computed initial value of the induction variable for the
  // GPU thread on which this being run.
  Value *ivBeg =
      builder.CreateIntCast(tipbdxbi, ivType, /*isSigned=*/false, "ivbeg.x");

  // The final value of the induction variable will be sum of the initial value
  // and the grainsize. In most cases, this will just be the `ivBeg.x + 1`.
  Value *ivEnd = builder.CreateAdd(ivBeg, getGrainsize(ivType), "ivend.x");

  // If the computed value of the induction variable for the given thread is
  // larger than the trip count, bypass the body of the loop.
  // FIXME: Don't assume the first argument here.
  Argument *tcX = f.getArg(1);
  Value *ivCond = builder.CreateICmpUGE(ivBeg, tcX);
  ReplaceInstWithInst(bbEntry->getTerminator(),
                      BranchInst::Create(bbExit, bbHeader, ivCond));

  // Otherwise, set the initial value of the loop to be this computed value.
  iv->getIncomingValueForBlock(bbEntry)->replaceAllUsesWith(ivBeg);

  // Then change the loop condition to check for the computed final value.
  ICmpInst *condX = cast<ICmpInst>(vmap[tl.getCondition()]);
  condX->setOperand(getOpIndex(*condX, tcX), ivEnd);
}

void GPUTTLoopBase::postProcessOutline(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                                       ValueToValueMapTy &vmap) {
  // The syncregion should not be present in the device module.
  Task *task = tl.getTask();
  Value *syncreg = vmap[task->getDetach()->getSyncRegion()];
  cast<Instruction>(syncreg)->eraseFromParent();

  Function *kernelF = toi.Outline;
  kernelF->setName(kernelName);
  kernelF->addFnAttr(Attribute::KitKernel);

  setKernelFuncAttrs(*kernelF);
  setKernelFuncCallingConv(*kernelF);
  setKernelFuncLinkage(*kernelF);
  setKernelFuncVisibility(*kernelF);
  setModuleAttrsForKernelFunc(*kernelF);

  processOutlinedIVs(*kernelF, tl, vmap);
}

void GPUTTLoopBase::processOutlinedLoopCall(TapirLoopInfo &tl,
                                            TaskOutlineInfo &toi,
                                            DominatorTree &dt) {
  LLVMContext &ctx = hostM.getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);

  CallBase *call = cast<CallBase>(toi.ReplCall);
  Constant *ctt = toConstant(tt, ctx);

  // The embedded bitcode global variable will have been created already. It
  // will not have been initialized, but we only need the address here.
  GlobalVariable *embFB = getEmbFBGlobal(tt, hostM);
  assert(embFB && "Require global variable for device code");

  // The name of the kernel function must be passed as a string to the runtime.
  Value *kName = createConstString(kernelName, hostM);

  // At this point we need a threads-per-block value for the launch call. There
  // are a number of ways that this can be set. The tapir loop may have an
  // explicit value attached to it (usually when an attribute is added to the
  // forall loop). This takes priority over a value provided on the command
  // line. Otherwise, set tpb to 0 to allow the runtime to pick an appropriate
  // value.
  unsigned tpbHint = getThreadsPerBlockAttr(*tl.getLoop()).value_or(0);
  unsigned fixedTPB = getOptions().getFixedThreadsPerBlock();
  Value *tpb = nullptr;
  if (tpbHint)
    tpb = ConstantInt::get(i32, tpbHint);
  else if (fixedTPB)
    tpb = ConstantInt::get(i32, fixedTPB);
  else
    tpb = ConstantInt::get(i32, 0);
  assert(tpb && "Threads per block cannot be null");

  // The trip counts will be the second, fourth and sixth arguments to the
  // outlined functions (depending on the depth of the tapir loop).
  Constant *zero = ConstantInt::get(i64, 0);
  Value *arg1 = call->getArgOperand(1);
  Value *arg3 = kernelDepth > 1 ? call->getArgOperand(3) : zero;
  Value *arg5 = kernelDepth > 2 ? call->getArgOperand(5) : zero;

  // Create a kernel properties global variable. This will be initialized in a
  // later pass. But for now, we only need it to exist.
  GlobalVariable *kProps = createKernelPropertiesGlobal(kernelName, tt, hostM);

  BasicBlock *bbNew = call->getParent()->splitBasicBlock(call);
  IRBuilder<> builder(&bbNew->front());

  // We need to explicitly sync non-const globals that are used in the kernel
  // before the kernel is launched.
  copyNonConstGlobalsHToD(builder);

  // The trip counts may be an argument or zero.
  Value *tcX = builder.CreateIntCast(arg1, i64, /*isSigned=*/false);
  Value *tcY = builder.CreateIntCast(arg3, i64, /*isSigned=*/false);
  Value *tcZ = builder.CreateIntCast(arg5, i64, /*isSigned=*/false);

  // Get or create a stream.
  Value *stream = builder.CreateIntrinsic(Intrinsic::kit_thread_stream, {ctt});

  SmallVector<Value *, 16> args = {
      ctt, embFB, kName, tcX, tcY, tcZ, tpb, kProps, stream,
  };
  for (Value *inp : call->args())
    args.push_back(inp);

  // TODO: We should probably have the launch and sync kitsune intrinsics take
  // a sync region as an argument This may make it easier to do post-outlining
  // analyses to eliminate/delay device synchronization calls instead of
  // always synchronizing immediately after the kernel launch.
  (void)builder.CreateIntrinsic(Intrinsic::kit_async_launch_kernel, args);

  // We explicitly add a sync here because the loop-spawning pass that drives
  // this tapir target does not call the lowerSync callback. If it did, this
  // could, correctly, be moved there.
  (void)builder.CreateIntrinsic(Intrinsic::kit_sync_stream, {ctt, stream});

  // After the kernel is done, copy the non-const globals back to the host. This
  // is done here to keep this part of the code generation simple. A subsequent
  // pass will attempt to move this call to the point where the global is
  // actually used on the host (or perhaps even delete it if the host never uses
  // the global again).
  copyNonConstGlobalsDToH(builder);

  call->eraseFromParent();
}
