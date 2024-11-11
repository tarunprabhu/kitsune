//==- LowerMobileIntrinsics.cpp - Lower kitsune mobile intrinsics -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers the Kitsune mobile intrinsics. For now, these are only the
// allocation and deallocation intrinsics, but these may be expanded to include
// explicit memory movement intrinsics as well.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Kitsune/LowerMobileIntrinsics.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

using namespace llvm;

#define DEBUG_TYPE "lower-mobile-intrinsics"

class LowerMobileAllocations {
private:
  TargetLibraryInfo &tli;

private:
  /// The allocator to use when lowering.
  enum class KitrtAllocator {
    None,    /// Don't use replace the intrinsic
    Cuda,    /// Use Kitsune's cuda allocator
    Default, /// Use Kitsune's default allocator
    Hip,     /// Use Kitsune's hip allocator
    System,  /// Use the system memory allocator. Don't use Kitsune's.
  };

private:
  /// TODO: Currently, this is very naive and simply looks at the tapir target
  /// set in the TargetLibraryInfo. This will not work correctly in multi-target
  /// mode. But that requires a more sophisticated analysis which should be
  /// implemented eventually.
  KitrtAllocator determineAllocatorKind(CallInst &) {
    switch (tli.getTapirTarget()) {
    case TapirTargetID::Cuda:
      return KitrtAllocator::Cuda;
    case TapirTargetID::Hip:
      return KitrtAllocator::Hip;
    case TapirTargetID::None:
      return KitrtAllocator::None;
    case TapirTargetID::Last_TapirTargetID:
      return KitrtAllocator::System;
    default:
      return KitrtAllocator::Default;
    }
  }

  /// Replace the call to a kitsune mobile allocation instruction with a call to
  /// an appropriate kitrt allocator. Returns true if the call was replaced,
  /// false otherwise.
  bool replace(CallInst &call) {
    // Do some (potentially not cheap) analysis to decide what would be a good
    // allocator to use here.
    KitrtAllocator kind = determineAllocatorKind(call);
    StringRef fname = "";
    switch (kind) {
    case KitrtAllocator::Cuda:
      fname = "__kitcuda_mem_alloc_managed";
      break;
    case KitrtAllocator::Hip:
      fname = "__kithip_mem_alloc_managed";
      break;
    case KitrtAllocator::Default:
      fname = "__kitrt_default_mem_alloc";
      break;
    case KitrtAllocator::System:
      fname = "malloc";
      break;
    case KitrtAllocator::None:
      return false;
    }

    Module &mod = *call.getModule();
    LLVMContext &ctxt = mod.getContext();
    Type *i64 = Type::getInt64Ty(ctxt);
    Type *ptr = call.getType();
    FunctionType *fty = FunctionType::get(ptr, {i64}, false);
    auto *fn = cast<Function>(mod.getOrInsertFunction(fname, fty).getCallee());

    fn->addRetAttr(Attribute::NoAlias);
    CallInst *newCall = CallInst::Create(fty, fn, {call.getArgOperand(0)},
                                         call.getName(), &call);
    newCall->addRetAttr(Attribute::NoAlias);
    call.replaceAllUsesWith(newCall);

    return true;
  }

public:
  LowerMobileAllocations(TargetLibraryInfo &tli) : tli(tli) {}

  /// Lower the mobile allocations in the given function. Return true if any
  /// mobile allocations were found and lowered, false otherwise.
  bool run(Function &f) {
    bool changed = false;
    std::vector<CallInst *> calls;
    for (inst_iterator i = inst_begin(f); i != inst_end(f); ++i) {
      if (auto *call = dyn_cast<CallInst>(&*i)) {
        if (call->getIntrinsicID() == Intrinsic::kitsune_mobile_alloc) {
          if (replace(*call)) {
            changed |= true;
            calls.push_back(call);
          }
        }
      }
    }
    for (CallInst *call : calls)
      call->eraseFromParent();
    return changed;
  }
};

class LowerMobileDeallocations {
private:
  TargetLibraryInfo &tli;

private:
  /// The deallocator to use when lowering.
  enum class KitrtDeallocator {
    None,    /// Don't replace the intrinsic
    Cuda,    /// Use Kitsune's cuda deallocator
    Default, /// Use the Kitsune's default deallocator
    Hip,     /// Use Kitsune's hip deallocator
    System,  /// Use the system's deallocator. Don't use Kitsune's
  };

private:
  /// TODO: Currently, this is very naive and simply looks at the tapir target
  /// set in the TargetLibraryInfo. This will not work correctly in multi-target
  /// mode. But that requires a more sophisticated analysis which should be
  /// implemented eventually.
  KitrtDeallocator determineDeallocatorKind(CallInst &) {
    switch (tli.getTapirTarget()) {
    case TapirTargetID::Cuda:
      return KitrtDeallocator::Cuda;
    case TapirTargetID::Hip:
      return KitrtDeallocator::Hip;
    case TapirTargetID::None:
      return KitrtDeallocator::None;
    case TapirTargetID::Last_TapirTargetID:
      return KitrtDeallocator::System;
    default:
      return KitrtDeallocator::Default;
    }
  }

  /// Replace the call to a kitsune mobile deallocation instruction with a call
  /// to an appropriate kitrt deallocator. Returns true if the call was
  /// replaced, false otherwise.
  bool replace(CallInst &call) {
    // Do some (potentially not cheap) analysis to decide what would be a good
    // allocator to use here.
    KitrtDeallocator kind = determineDeallocatorKind(call);
    StringRef fname = "";
    switch (kind) {
    case KitrtDeallocator::Cuda:
      fname = "__kitcuda_mem_free";
      break;
    case KitrtDeallocator::Hip:
      fname = "__kithip_mem_free";
      break;
    case KitrtDeallocator::Default:
      fname = "__kitrt_default_mem_free";
      break;
    case KitrtDeallocator::System:
      fname = "free";
      break;
    case KitrtDeallocator::None:
      return false;
    }

    Module &mod = *call.getModule();
    LLVMContext &ctxt = mod.getContext();
    Type *voidTy = Type::getVoidTy(ctxt);
    Type *ptr = call.getArgOperand(0)->getType();
    FunctionType *fty = FunctionType::get(voidTy, {ptr}, false);
    auto *fn = cast<Function>(mod.getOrInsertFunction(fname, fty).getCallee());

    CallInst *newCall = CallInst::Create(fty, fn, {call.getArgOperand(0)},
                                         call.getName(), &call);
    call.replaceAllUsesWith(newCall);

    return true;
  }

public:
  LowerMobileDeallocations(TargetLibraryInfo &tli) : tli(tli) {}

  /// Lower the mobile allocations in the given function. Return true if any
  /// mobile allocations were found and lowered, false otherwise.
  bool run(Function &f) {
    bool changed = false;
    std::vector<CallInst *> calls;
    for (inst_iterator i = inst_begin(f); i != inst_end(f); ++i) {
      if (auto *call = dyn_cast<CallInst>(&*i)) {
        if (call->getIntrinsicID() == Intrinsic::kitsune_mobile_free) {
          if (replace(*call)) {
            changed |= true;
            calls.push_back(call);
          }
        }
      }
    }
    for (CallInst *call : calls)
      call->eraseFromParent();
    return changed;
  }
};

PreservedAnalyses LowerMobileIntrinsicsPass::run(Module &m,
                                                 ModuleAnalysisManager &mam) {
  // We may need to do some analysis here, or call an analysis pass before we
  // start modifying the functions.
  for (Function &f : m) {
    FunctionAnalysisManager &fam =
        mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();
    TargetLibraryInfo &tli = fam.getResult<TargetLibraryAnalysis>(f);

    LowerMobileAllocations(tli).run(f);
    LowerMobileDeallocations(tli).run(f);
  }

  // For the allocations and deallocations, we simply replace function calls
  // with others that accept exactly the same arguments and which have exactly
  // the same attributes. None of the analyses should be invalidated as a
  // result.
  //
  // TODO: If we ever have to support explicit memory movement intrinsics, we
  // may need to revisit this. There is a possibility that it is not safe to
  // assume that all analyses will remain valid in that case.
  return PreservedAnalyses::all();
}
