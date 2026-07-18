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
#include "kitsune/Core/FuncAttrs.h"
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/KernelProperties.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Core/ValueUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

static const StringRef suffixes[3] = {".x", ".y", ".z"};

GPUTTLoopBase::GPUTTLoopBase(Module &hostM, Module &devM, const TTOptions &tto,
                             const TapirLoopInfo &tl, TTID tt, StringRef name)
    : LoopOutlineProcessor(hostM, devM, tto,
                           CloneFunctionChangeType::DifferentModule),
      devM(devM), hostM(hostM), loops({tl.getLoop()}), tt(tt),
      kernelName(name) {
  Loop *root = tl.getLoop();
  unsigned depth = getPerfectDepthAttr(*root).value_or(0);
  assert(depth >= 1 && depth <= 3 &&
         "Depth of loop lowered by GPU must be in the range [1,3]");

  // We have already added the root of the loop nest being lowered to the
  // loops member. Now, add the children. Only after this loop is it safe to use
  // the getDepth() method.
  for (unsigned i = 1; i < depth; ++i)
    loops.push_back(loops.back()->getSubLoops().front());

  // Sanity check the loops. These should have been enforced before we get here,
  // but check again in case something changes upstream.
  for (unsigned i = 0; i < depth; ++i) {
    Loop *loop = loops[i];
    assert(loop->isLoopSimplifyForm() &&
           "All loops in a tapir loop nest must be in simplify form");
    assert(loop->getCanonicalInductionVariable() &&
           "All loops in a tapir loop nest must be canonical");
    assert(loop->getLatchCmpInst() &&
           "Could not get loop latch compare instruction");

    // The loop nest lowered using this tapir target is expected to be perfect.
    // Each loop being lowered should have exactly one child. The innermost loop
    // in the nest may have more than one subloops because those will not be
    // lowered.
    if (i < depth - 1)
      assert(loop->getSubLoops().size() == 1 && "Expecting only 1 subloop");
  }
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
          ptrTy, Intrinsic::kit_gpu_symbol_address, {ctt, fb, name});
      if (copyFn == Intrinsic::kit_gpu_symbol_memcpy_dtoh)
        (void)builder.CreateIntrinsic(voidTy, copyFn, {ctt, g, devPtr, bytes});
      else if (copyFn == Intrinsic::kit_gpu_symbol_memcpy_htod)
        (void)builder.CreateIntrinsic(voidTy, copyFn, {ctt, devPtr, g, bytes});
      else
        llvm_unreachable("copyNonConstGlobals: Invalid intrinsic");
    }
  }
}

void GPUTTLoopBase::copyNonConstGlobalsDToH(IRBuilder<> &builder) {
  copyNonConstGlobals(builder, Intrinsic::kit_gpu_symbol_memcpy_dtoh);
}

void GPUTTLoopBase::copyNonConstGlobalsHToD(IRBuilder<> &builder) {
  copyNonConstGlobals(builder, Intrinsic::kit_gpu_symbol_memcpy_htod);
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
        addDeviceAttr(*devf);
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

void GPUTTLoopBase::serializeKernelFunc(Function &f) {
  auto shouldRemove = [](const Instruction &inst) -> bool {
    if (isa<DetachInst>(inst) || isa<ReattachInst>(inst) || isa<SyncInst>(inst))
      return true;
    else if (const auto *call = dyn_cast<CallBase>(&inst))
      if (Intrinsic::ID callee = call->getIntrinsicID())
        return callee == Intrinsic::syncregion_start;
    return false;
  };

  // Collect the instructions to be deleted.
  SmallVector<Instruction *, 8> del;
  for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
    if (shouldRemove(*i))
      del.push_back(cast<Instruction>(&*i));

  // Fixup the terminator instructions.
  for (Instruction *inst : del) {
    if (auto *detach = dyn_cast<DetachInst>(inst))
      ReplaceInstWithInst(detach, BranchInst::Create(detach->getDetached()));
    else if (auto *reattach = dyn_cast<ReattachInst>(inst))
      ReplaceInstWithInst(reattach,
                          BranchInst::Create(reattach->getDetachContinue()));
    else if (auto *sync = dyn_cast<SyncInst>(inst))
      ReplaceInstWithInst(sync, BranchInst::Create(sync->getSuccessor(0)));
  }

  // Finally, delete the syncregion creation calls.
  for (Instruction *inst : del)
    if (isa<CallInst>(inst))
      inst->eraseFromParent();
}

void GPUTTLoopBase::updateTripCount(Loop *loop, PHINode *iv, Value *newTC,
                                    const ValueToValueMapTy &vmap) {
  // For the outermost loop in a nest that is being lowered, we know what the
  // expected trip count is because it will be passed in as an argument to the
  // generated kernel function. However, because tapir does not know about the
  // inner loops in the nest, the trip counts on those will not have been
  // wired up to the corresponding kernel function arguments (in effect, we
  // have generated a function with unused arguments). As a result, we do not
  // know what the expected trip count is in compare instruction in the loop
  // latch.
  //
  // Therefore, assume that the operand of the latch that is *not* the updated
  // loop induction variable is the trip count.
  BasicBlock *be = getUniqueBackEdge(*loop);
  BasicBlock *backEdge = cast<BasicBlock>(vmap.lookup(be));
  Value *incr = iv->getIncomingValueForBlock(backEdge);
  ICmpInst *latchCmp = cast<ICmpInst>(vmap.lookup(loop->getLatchCmpInst()));

  replaceNonMatchingOperands(*latchCmp, incr, newTC);
}

GlobalVariable *GPUTTLoopBase::getDevGlobal(GlobalVariable *g,
                                            const ValueToValueMapTy &vmap) {
  return cast<GlobalVariable>(stripCasts(cast<Constant>(vmap.lookup(g))));
}

unsigned GPUTTLoopBase::getDepth() const { return loops.size(); }

void GPUTTLoopBase::setKernelFuncLinkage(Function &f) {
  f.setLinkage(GlobalValue::ExternalLinkage);
}

void GPUTTLoopBase::setupLoopControlArgs(TapirLoopInfo *tl,
                                         SmallVectorImpl<Value *> &lcArgs,
                                         SmallVectorImpl<Value *> &lcInputs) {
  auto isIncr = [](Value *v, PHINode *iv) -> bool {
    Type *ty = v->getType();
    if (auto *inst = dyn_cast<Instruction>(v))
      if (inst->getOpcode() == Instruction::Add)
        if ((isIntOne(inst->getOperand(0), ty) && inst->getOperand(1) == iv) ||
            (isIntOne(inst->getOperand(1), ty) && inst->getOperand(0) == iv))
          return true;
    return false;
  };

  // This tries to get the trip count from the compare instruction in the
  // latch. One of the operands of the instruction must be an increment - the
  // other, the trip count. This only works because we require the tapir loops
  // that are being lowered to have unique, canonical induction variables.
  //
  // FIXME: It would be good if we could use some of the methods available in
  // in the loop class to find these. But those require analysis results that
  // are not available to this object. Making them available may also be tricky
  // since they may have been invalidated by the transformations of other loops
  // in the function.
  //
  // One possible way to do this may be to pre-compute the loop bounds for the
  // loops and have those be accessible somewhere.
  //
  auto getTripCount = [&isIncr](ICmpInst *cmp, PHINode *iv) -> Value * {
    Value *op0 = cmp->getOperand(0);
    Value *op1 = cmp->getOperand(1);
    if (isIncr(op0, iv))
      return op1;
    else if (isIncr(op1, iv))
      return op0;
    llvm_unreachable("getTripCount: Didn't find induction variable");
  };

  assert(tl->getInductionVars()->size() == 1 &&
         "Tapir loop must have a single primary induction variable");

  // Iterate over the loops in reverse order because the arguments to the
  // kernel function are always in order from innermost to outermost.
  for (unsigned i = 0; i < getDepth(); ++i) {
    Loop *loop = loops[i];
    BasicBlock *ph = loop->getLoopPreheader();
    ICmpInst *cmp = loop->getLatchCmpInst();
    PHINode *iv = loop->getCanonicalInductionVariable();
    Dirxn dirxn = dirxns[getDepth() - i - 1];
    StringRef sfx = suffixes[int(dirxn)];

    // Since the start value is 0, we don't strictly need this. However, not
    // passing this causes issues in loop spawning since that assumes that this
    // value will be passed. The fixes needed to make this work in loop spawning
    // are not particularly difficult, but it does feel messy. For now, we just
    // pass it since the fix to loop spawning will likely require some more
    // thought.
    Value *ivBeg = iv->getIncomingValueForBlock(ph);
    std::string nameBeg = join_items("", "zero", sfx);
    LoopCtlArgs.push_back(new Argument(ivBeg->getType(), nameBeg));
    lcArgs.push_back(LoopCtlArgs.back());
    lcInputs.push_back(ivBeg);

    Value *ivEnd = getTripCount(cmp, iv);
    std::string nameEnd = join_items("", "tc", sfx);
    LoopCtlArgs.push_back(new Argument(ivEnd->getType(), nameEnd));
    lcArgs.push_back(LoopCtlArgs.back());
    lcInputs.push_back(ivEnd);
  }
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

void GPUTTLoopBase::emitIndexCalculation(IRBuilder<> &builder, PHINode *iv,
                                         Dirxn dirxn,
                                         SmallVector<IVRange, 4> &ivRanges) {
  // clang-format off
  auto getThreadIdFn = [&dirxn]() -> Intrinsic::ID {
    switch (dirxn) {
    case Dirxn::X: return Intrinsic::kit_gpu_thread_id_x;
    case Dirxn::Y: return Intrinsic::kit_gpu_thread_id_y;
    case Dirxn::Z: return Intrinsic::kit_gpu_thread_id_z;
    }
    llvm_unreachable("getThreadIdFn: Dirxn not handled");
  };

  auto getBlockIdFn = [&dirxn]() -> Intrinsic::ID {
    switch (dirxn) {
    case Dirxn::X: return Intrinsic::kit_gpu_block_id_x;
    case Dirxn::Y: return Intrinsic::kit_gpu_block_id_y;
    case Dirxn::Z: return Intrinsic::kit_gpu_block_id_z;
    }
    llvm_unreachable("getBlockIdFn: Dirxn not handled");
  };

  auto getBlockSizeFn = [&dirxn]() -> Intrinsic::ID {
    switch (dirxn) {
    case Dirxn::X: return Intrinsic::kit_gpu_block_size_x;
    case Dirxn::Y: return Intrinsic::kit_gpu_block_size_y;
    case Dirxn::Z: return Intrinsic::kit_gpu_block_size_z;
    }
    llvm_unreachable("getBlockSizeFn: Dirxn not handled");
  };
  // clang-format on

  // Construct a name with the given base and suffixing the direction. These are
  // only intended for convenience if we ever have to read the IR.
  auto n = [&dirxn](StringRef base) -> std::string {
    return join_items("", base, suffixes[int(dirxn)]);
  };

  LLVMContext &ctx = builder.getContext();

  // The outlined loop runs from [iv0, tc] where iv0 and tc are bounds passed to
  // the kernel function. Convert these to use threadIdx, blockIdx, blockDim
  // etc.
  //
  // This is the classic calculation for the induction variable i:
  //
  //     i = blockDim.[[D]] * blockIdx.[[D]] + threadId.[[D]]
  //
  // where [[D]] must be one of 'x', 'y', or 'z'.
  //
  // The calculation below assumes that iv0.[[D]] == 0.This is enforced earlier
  // in the lowering process and is unlikely to ever change. If this is ever
  // non-zero, it will likely cause a lot of problems everywhere.
  //
  Value *ctt = toConstant(tt, ctx);
  Value *tid = builder.CreateIntrinsic(getThreadIdFn(), {ctt}, /*FMFSource=*/{},
                                       n("tid"));
  Value *bid = builder.CreateIntrinsic(getBlockIdFn(), {ctt}, /*FMFSource=*/{},
                                       n("bid"));
  Value *bsz = builder.CreateIntrinsic(getBlockSizeFn(), {ctt},
                                       /*FMFSource=*/{}, n("bsz"));
  Value *bdxbi = builder.CreateMul(bsz, bid);
  Value *bdxbipti = builder.CreateAdd(bdxbi, tid, n(".ivb"));

  // This is the computed initial value of the induction variable for the
  // GPU thread on which this being run.
  Type *ivType = iv->getType();
  Value *ivBeg =
      builder.CreateIntCast(bdxbipti, ivType, /*isSigned=*/false, n("ivb"));

  // The final value of the induction variable will be just be the `ivBeg + 1`.
  Value *one = ConstantInt::get(ivType, 1, /*isSigned=*/false);
  Value *ivEnd = builder.CreateAdd(ivBeg, one, n("ive"));

  ivRanges.push_back({iv, ivBeg, ivEnd});
}

void GPUTTLoopBase::processOutlinedIVs(Function &f, TapirLoopInfo &tl,
                                       const ValueToValueMapTy &vmap) {
  Loop *root = loops.front();
  BasicBlock *bbEntry = cast<BasicBlock>(vmap.lookup(root->getLoopPreheader()));
  IRBuilder<> builder(bbEntry->getTerminator());

  // Compute the new ranges of the induction variables of all loops in the nest.
  SmallVector<IVRange, 4> ivRanges;
  for (unsigned i = 0; i < getDepth(); ++i) {
    Loop *loop = loops[i];
    PHINode *iv =
        cast<PHINode>(vmap.lookup(loop->getCanonicalInductionVariable()));
    Dirxn dirxn = dirxns[getDepth() - i - 1];

    emitIndexCalculation(builder, iv, dirxn, ivRanges);
  }

  // Collect the trip counts for all loops in the next. These are passed in as
  // arguments to the kernel function. The argument lists for nests with
  // various depths are summarized below:
  //
  //  .----------------------------------------------------------------------.
  //  | Depth |                      Arguments                               |
  //  |----------------------------------------------------------------------|
  //  |   1   | i64 z.x, i64 tc.x, ...                                       |
  //  |   2   | i64 z.y, i64 tc.y, i64 z.x, i64 tc.x, ...                    |
  //  |   3   | i64 z.z, i64 tc.z, i64 z.y, i64 tc.y, i64 z.x, i64 tc.x, ... |
  //  '----------------------------------------------------------------------'
  //
  // Here, z.x, z.y, and z.z are all expected to be 0. The argument names
  // suffixed with .x are intended for the innermost loop, .y for the parent
  // of the innermost loop, and .z for the grandparent of the innermost loop.
  //
  SmallVector<Argument *> tcs;
  for (unsigned i = 0; i < getDepth(); ++i)
    tcs.push_back(f.getArg(2 * i + 1));

  // Check that the start of all induction variables are less than the
  // corresponding trip counts.
  SmallVector<Value *, 4> cmps;
  for (unsigned i = 0; i < getDepth(); ++i) {
    const IVRange &ivRange = ivRanges[i];
    Value *ivBeg = ivRange.beg;
    Value *tc = tcs[i];
    Value *cmp = builder.CreateICmpULT(ivBeg, tc);

    cmps.push_back(cmp);
  }

  // If all the induction variables are in range, enter the loop, otherwise,
  // bypass it altogether.
  Value *allInRange = builder.CreateAnd(cmps);
  BasicBlock *bbHeader = cast<BasicBlock>(vmap.lookup(root->getHeader()));
  BasicBlock *bbExit = cast<BasicBlock>(vmap.lookup(tl.getExitBlock()));
  ReplaceInstWithInst(bbEntry->getTerminator(),
                      BranchInst::Create(bbHeader, bbExit, allInRange));

  // Otherwise, set the initial values of the loop induction variables to be
  // these computed values.
  for (unsigned i = 0; i < getDepth(); ++i) {
    const IVRange &ivRange = ivRanges[i];
    PHINode *iv = ivRange.iv;
    Value *ivBeg = ivRange.beg;
    Loop *loop = loops[i];
    BasicBlock *ph = cast<BasicBlock>(vmap.lookup(loop->getLoopPreheader()));

    iv->setIncomingValueForBlock(ph, ivBeg);
  }

  // Then change the loop conditions to check for the computed final values.
  for (unsigned i = 0; i < getDepth(); ++i) {
    Loop *loop = loops[i];
    PHINode *iv = ivRanges[i].iv;
    Value *ivEnd = ivRanges[i].end;

    updateTripCount(loop, iv, ivEnd, vmap);
  }
}

void GPUTTLoopBase::postProcessOutline(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                                       ValueToValueMapTy &vmap) {
  // The syncregion should not be present in the device module.
  Task *task = tl.getTask();
  Value *syncreg = vmap[task->getDetach()->getSyncRegion()];
  cast<Instruction>(syncreg)->eraseFromParent();

  Function *kernelF = toi.Outline;
  kernelF->setName(kernelName);
  addKernelAttr(*kernelF);

  setKernelFuncAttrs(*kernelF);
  setKernelFuncCallingConv(*kernelF);
  setKernelFuncLinkage(*kernelF);
  setKernelFuncVisibility(*kernelF);
  setModuleAttrsForKernelFunc(*kernelF);

  serializeKernelFunc(*kernelF);

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

  // At this point we need a threads-per-block value for the launch call. If the
  // tapir loop has an explicit value attached to it, use that, otherwise, set
  // it to zero. This tells the runtime to compute a value to use.
  Loop *loop = tl.getLoop();
  Value *tpb = ConstantInt::get(i32, getThreadsPerBlockAttr(*loop).value_or(0));

  // The trip counts will be the second, fourth and sixth arguments to the
  // outlined functions (depending on the depth of the tapir loop).
  Constant *zero = ConstantInt::get(i64, 0);
  Value *argX = zero;
  Value *argY = zero;
  Value *argZ = zero;
  switch (getDepth()) {
  case 1:
    argX = call->getArgOperand(1);
    break;
  case 2:
    argY = call->getArgOperand(1);
    argX = call->getArgOperand(3);
    break;
  case 3:
    argZ = call->getArgOperand(1);
    argY = call->getArgOperand(3);
    argX = call->getArgOperand(5);
    break;
  default:
    llvm_unreachable("Unexpected depth of tapir loop nest");
  }

  // Create a kernel properties global variable. This will be initialized in a
  // later pass. But for now, we only need it to exist.
  GlobalVariable *kProps = createKernelPropertiesGlobal(kernelName, tt, hostM);

  BasicBlock *bbNew = call->getParent()->splitBasicBlock(call);
  IRBuilder<> builder(&bbNew->front());

  // We need to explicitly sync non-const globals that are used in the kernel
  // before the kernel is launched.
  copyNonConstGlobalsHToD(builder);

  // The trip counts may be an argument or zero.
  Value *tcX = builder.CreateIntCast(argX, i64, /*isSigned=*/false);
  Value *tcY = builder.CreateIntCast(argY, i64, /*isSigned=*/false);
  Value *tcZ = builder.CreateIntCast(argZ, i64, /*isSigned=*/false);

  // Get or create a stream.
  Value *stream = builder.CreateIntrinsic(Intrinsic::kit_gpu_stream_new, {ctt});

  SmallVector<Value *, 16> args = {
      ctt, embFB, kName, tcZ, tcY, tcX, tpb, kProps, stream,
  };
  for (Value *inp : call->args())
    args.push_back(inp);

  // TODO: We should probably have the launch and sync kitsune intrinsics take
  // a sync region as an argument This may make it easier to do post-outlining
  // analyses to eliminate/delay device synchronization calls instead of
  // always synchronizing immediately after the kernel launch.
  (void)builder.CreateIntrinsic(Intrinsic::kit_async_gpu_kernel_launch, args);

  // We explicitly add a sync here because the loop-spawning pass that drives
  // this tapir target does not call the lowerSync callback. If it did, this
  // could, correctly, be moved there.
  (void)builder.CreateIntrinsic(Intrinsic::kit_gpu_stream_sync, {ctt, stream});

  // After the kernel is done, copy the non-const globals back to the host. This
  // is done here to keep this part of the code generation simple. A subsequent
  // pass will attempt to move this call to the point where the global is
  // actually used on the host (or perhaps even delete it if the host never uses
  // the global again).
  copyNonConstGlobalsDToH(builder);

  call->eraseFromParent();
}
