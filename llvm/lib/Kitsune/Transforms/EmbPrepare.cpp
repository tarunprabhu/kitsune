//===- EmbPrepare.cpp - Prepare embedded modules for codegen --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prepare embedded modules for code generation.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EmbPrepare.h"
#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/Core/TapirTargetOptions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/AMDGPUAddrSpace.h"

#define DEBUG_TYPE "emb-prepare"

using namespace llvm;

static cl::opt<bool> clInlineAll(
    "emb-inline-all", cl::init(false),
    cl::desc("Inline all device functions in the kernel module, unless they "
             "have the 'noinline' attribute"),
    cl::NotHidden);

static cl::opt<bool> clInlineAllForce(
    "emb-inline-all-force", cl::init(false),
    cl::desc("Inline all device functions in the kernel module, including "
             "those that have the 'noinline' attribute"));

namespace {

class EmbPrepareCuda {
private:
  const TapirTargetOptions &tto;

private:
  /// Fix the attributes on the "non-kernel" functions. The attributes on the
  /// kernel function will have been set by the tapir targets.
  bool fixDeviceFuncAttrs(Module &devM) {
    bool changed = false;
    for (Function &f : devM.functions()) {
      if (f.hasFnAttribute(Attribute::KitDevice)) {
        f.removeFnAttr("target-cpu");
        f.removeFnAttr("target-features");
        f.removeFnAttr("tune-cpu");

        f.addFnAttr("target-cpu", tto.getCudaArch());
        f.addFnAttr(
            "target-features",
            join_items(",", tto.getCudaTargetFeatures(), tto.getCudaArch()));

        bool hasNoInline = f.hasFnAttribute(Attribute::NoInline);
        if (clInlineAllForce) {
          f.removeFnAttr(Attribute::NoInline);
          f.addFnAttr(Attribute::AlwaysInline);
        } else if (clInlineAll and (not hasNoInline)) {
          f.addFnAttr(Attribute::AlwaysInline);
        }

        changed |= true;
      }
    }
    return changed;
  }

public:
  EmbPrepareCuda(const TapirTargetOptions &tto) : tto(tto) {}

  bool run(Module &devM) {
    bool changed = false;

    changed |= fixDeviceFuncAttrs(devM);

    return changed;
  }
};

class EmbPrepareHip {
private:
  const TapirTargetOptions &tto;

private:
  /// Fix the address space of all pointer arguments to the kernels.
  ///
  /// FIXME: This makes extensive use of mutateType in order to fix things up.
  /// This is far from ideal and goes against "the LLVM way". However, any
  /// other approach would probably require a custom cloning utility that
  /// transforms types when outlining from the tapir loop to the kernel
  /// function. That is likely to be a considerable amount of work. For now,
  /// this "works", so we'll stick with it, but it would be nice if we could
  /// avoid having to do such things.
  bool fixKernelArgumentAddrSpace(Module &devM) {
    bool changed = false;
    LLVMContext &ctx = devM.getContext();
    PointerType *ptrTy = PointerType::getUnqual(ctx);
    PointerType *ptrASTy = PointerType::get(ctx, AMDGPUAS::GLOBAL_ADDRESS);

    // The implementation of this function is truly appalling!
    // We will eventually mutate the types of those function arguments that are
    // pointers. However, the users of these arguments do not expect a pointer
    // in a non-default address space. In some cases, mutating the type of the
    // argument will result in a broken module; for instance, when the argument
    // passed to a function in a call instruction. To get around this, we cast
    // away the address space early in the function body and replace all uses of
    // the argument with the cast. This must be done in a series of carefully
    // orchestrated steps to avoid tripping up assertions and "optimizations"
    // (such as CastInst constructors not allowing the creation of "no-op" casts
    // in some cases) elsewhere in LLVM. The details are in comments in the body
    // of the loop below.
    for (Function &f : devM.functions()) {
      if (not f.hasFnAttribute(Attribute::KitKernel))
        continue;

      // If none of the function arguments are pointers, move on.
      if (std::none_of(f.arg_begin(), f.arg_end(), [](Argument &arg) -> bool {
            return isa<PointerType>(arg.getType());
          }))
        continue;

      // We need to create an instruction that casts away the address space
      // in the function arguments that are pointers (the arguments have not
      // been mutated yet). The cast instruction should be added to the entry
      // block of the function after the last alloca.
      IRBuilder builder(ctx);
      builder.SetInsertPointPastAllocas(&f);

      // We will need to create a new function type. Collect all the parameter
      // types as we are iterating over the arguments.
      std::vector<Type *> paramTys;
      for (Argument &arg : f.args()) {
        if (auto *argTy = dyn_cast<PointerType>(arg.getType())) {
          if (argTy->getAddressSpace() != AMDGPUAS::GLOBAL_ADDRESS) {
            assert(argTy->getAddressSpace() == 0 &&
                   "Argument must be in the default address space");

            // We cannot create a cast instruction before mutating the type of
            // the argument since the source and destination types will be the
            // same. Therefore, mutate the type of the argument.
            arg.mutateType(ptrASTy);

            // The cast can now be created since it will be valid.
            Value *cst = builder.CreateAddrSpaceCast(&arg, ptrTy);

            // We want to replace all uses of the argument with this cast, but
            // we cannot do so because the type of the argument and the cast
            // will be different. Therefore, restore the original type of the
            // argument.
            arg.mutateType(argTy);

            // We cannot use replaceAllUsesWith because that would replace the
            // operand to the cast instruction that we just created. Replace
            // everything except the cast.
            arg.replaceUsesWithIf(
                cst, [&](Use &u) -> bool { return u.getUser() != cst; });

            // Now that we have replaced all the uses (except the cast), we can
            // mutate the type once and for all.
            arg.mutateType(ptrASTy);
          }
        }

        // We need to create a new function type, so keep track of the types
        // of all the arguments. We must add the argument type here because it
        // could have changed in the truly appalling code above.
        paramTys.push_back(arg.getType());
      }

      Type *retTy = f.getReturnType();
      bool isVarArg = f.isVarArg();
      f.mutateValueType(FunctionType::get(retTy, paramTys, isVarArg));

      changed |= true;
    }

    return changed;
  }

  /// Fix the address space of all allocas in the device module. These must be
  /// in the alloca address space. This works by creating new alloca
  /// instructions in the alloca address space, then casting away the address
  /// space from the alloca and replacing all uses of the old alloca with this
  /// cast.
  bool fixAllocaAddrSpace(Module &devM) {
    bool changed = false;
    LLVMContext &ctx = devM.getContext();
    const DataLayout &dl = devM.getDataLayout();
    unsigned addrSpaceAlloca = dl.getAllocaAddrSpace();

    for (Function &f : devM.functions()) {
      std::vector<AllocaInst *> toFix;
      for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
        if (auto ai = dyn_cast<AllocaInst>(&*i))
          if (ai->getAddressSpace() != addrSpaceAlloca)
            toFix.push_back(ai);

      if (toFix.empty())
        continue;

      // The cast instructions must be added after all the allocas. The allocas
      // that replace the original allocas will be added immediately before the
      // ones that they are replacing.
      IRBuilder builder(ctx);
      builder.SetInsertPointPastAllocas(&f);

      for (AllocaInst *ai : toFix) {
        // The new alloca is created just before the alloca that it is
        // replacing.
        AllocaInst *newAlloca = new AllocaInst(
            ai->getAllocatedType(), addrSpaceAlloca, ai->getArraySize(),
            ai->getAlign(), ai->getName(), /*insertBefore=*/ai->getIterator());

        // The address space cast that strips away the address space is added
        // after the allocas.
        Value *cst = builder.CreateAddrSpaceCast(newAlloca, ai->getType());

        ai->replaceAllUsesWith(cst);
        ai->eraseFromParent();
      }

      changed |= true;
    }
    return changed;
  }

  /// Fix the attributes on the "non-kernel" functions. These may still have
  /// attributes that are only relevant if they are run on the CPU. The
  /// attributes on the kernel function will have been set by the tapir targets.
  bool fixDeviceFuncAttrs(Module &devM) {
    bool changed = false;
    for (Function &f : devM.functions()) {
      if (f.hasFnAttribute(Attribute::KitDevice)) {
        f.removeFnAttr("target-cpu");
        f.removeFnAttr("target-features");
        f.removeFnAttr("tune-cpu");
        f.removeFnAttr(Attribute::UWTable);

        f.addFnAttr(Attribute::NoUnwind);
        f.addFnAttr("target-cpu", tto.getHipArch());
        f.addFnAttr("target-features", tto.getHipTargetFeatures());

        bool hasNoInline = f.hasFnAttribute(Attribute::NoInline);
        if (clInlineAllForce) {
          f.removeFnAttr(Attribute::NoInline);
          f.addFnAttr(Attribute::AlwaysInline);
        } else if (clInlineAll and (not hasNoInline)) {
          f.addFnAttr(Attribute::AlwaysInline);
        }

        changed |= true;
      }
    }
    return changed;
  }

  /// Fix the calling convention on device functions and the callsites.
  bool fixCallingConventions(Module &devM) {
    bool changed = false;

    // Set the calling convention to fast on the device functions because
    // that is what hipcc does.
    for (Function &f : devM.functions()) {
      if (f.hasFnAttribute(Attribute::KitDevice)) {
        if (f.getCallingConv() != CallingConv::Fast) {
          f.setCallingConv(CallingConv::Fast);
          changed |= true;
        }
      }
    }

    for (Function &f : devM.functions())
      for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
        if (auto *ci = dyn_cast<CallBase>(&*i))
          if (Function *cf = ci->getCalledFunction())
            if (ci->getCallingConv() != cf->getCallingConv()) {
              ci->setCallingConv(cf->getCallingConv());
              changed |= true;
            }
    return changed;
  }

public:
  EmbPrepareHip(const TapirTargetOptions &tto) : tto(tto) {}

  bool run(Module &devM) {
    bool changed = false;

    changed |= fixKernelArgumentAddrSpace(devM);
    changed |= fixAllocaAddrSpace(devM);
    changed |= fixDeviceFuncAttrs(devM);
    changed |= fixCallingConventions(devM);

    return changed;
  }
};

} // namespace

namespace llvm {

bool EmbPreparePass::run(TTID tt, Module &devM, Module &hostM,
                         ModuleAnalysisManager &hostMAM) {
  const TapirTargetInfo &tgi = hostMAM.getResult<TapirTargetAnalysis>(hostM);
  const TapirTargetOptions &tto = tgi.getOptions();
  switch (tt) {
  case TTID::Cuda:
    return EmbPrepareCuda(tto).run(devM);
  case TTID::Hip:
    return EmbPrepareHip(tto).run(devM);
  default:
    llvm_unreachable("EmbPreparePass::run: TTID not handled");
  }
}

} // namespace llvm
