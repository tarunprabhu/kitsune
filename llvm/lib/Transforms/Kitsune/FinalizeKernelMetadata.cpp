//==- FinalizeKernelMetadata.cpp - Compute kernel metadata in launch calls -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compute the kernel metadata used in kernel launches.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Kitsune/FinalizeKernelMetadata.h"
#include "kitsune/Common/Types.h"
#include "llvm/Analysis/TapirTargetAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/KitsuneMetadata.h"
#include "llvm/Transforms/Utils/KitsuneUtils.h"

#include <map>
#include <set>

#define DEBUG_TYPE "finalize-kernel-metadata"

using namespace llvm;

/// Helper class to calculate the instruction kinds in a function. We need this
/// for two reasons:
///
///   1. To avoid redundant computations of the instruction mix of functions
///      that have already been encountered. These are functions that are
///      reachable from the kernel function.
///
///   2. To help ensure that we don't fall into an infinite recursion in case
///      the callgraph contains recursive functions.
///
class InstMixCalculator {
private:
  /// The instruction mixes for the functions seen so far. The value here is a
  /// unique pointer so we can return a reference to the object instead of
  /// returning by value. We need a unique_ptr in this case, otherwise, adding
  /// another pair to the map risks invalidating the reference.
  ///
  /// NOTE: The values here are for just the function alone, not including the
  /// the values of the instructions mixes for any functions reachable from it.
  /// This trades off a bit of efficiency but keeps the implementation simple.
  std::map<const Function *, std::unique_ptr<KernelInstMixData>> instMixes;

private:
  /// Find the functions with definitions that are reachable from the given
  /// function. A set of functions that have already been seen are provided to
  /// ensure that recursive functions (self, or mutual) are only processed once.
  void reachable(const Function &f, std::set<const Function *> &seen) {
    // insert returns a pair, the second element of which is true if insertion
    // took place, false otherwise. If f has already been seen, the second
    // element will be false indicating that the function has already been seen
    // and we should not process it.
    if (not seen.insert(&f).second)
      return;

    std::set<const Function *> calledFuncs;
    for (const_inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i) {
      if (auto *call = dyn_cast<CallBase>(&*i))
        // In some cases, we see inline assembly being called here, especially
        // when code from device-specific bitcode modules is inlined, so don't
        // assume that we will always get a function.
        if (Function *cf = call->getCalledFunction())
          if (cf->size())
            calledFuncs.insert(cf);
    }

    for (const Function *cf : calledFuncs)
      reachable(*cf, seen);
  }

  std::set<const Function *> reachable(const Function &f) {
    std::set<const Function *> seen;
    reachable(f, seen);
    return seen;
  }

  const KernelInstMixData &calculate(const Function &f) {
    if (instMixes.find(&f) == instMixes.end()) {
      auto instMix = std::make_unique<KernelInstMixData>();
      for (const_inst_iterator i = inst_begin(f), e = inst_end(f); i != e;
           ++i) {
        // For now, we only treat certain instructions as memory ops. This may
        // need to be updated depending on how sophisticated we wish to make
        // this calculator.
        if (isa<LoadInst>(*i) or isa<StoreInst>(*i) or isa<AtomicRMWInst>(*i) or
            isa<AtomicCmpXchgInst>(*i)) {
          instMix->memOps += 1;
        } else if (i->isUnaryOp() or i->isBinaryOp()) {
          if (i->getType()->isFPOrFPVectorTy())
            instMix->fpOps += 1;
          else if (i->getType()->isIntegerTy())
            instMix->intOps += 1;
          else
            instMix->otherOps += 1;
        }
      }
      instMixes.emplace(&f, std::move(instMix));
    }
    return *instMixes.at(&f);
  }

public:
  /// Calculate the instruction mix for the given function. The result will be
  /// the sum of the instruction mixes of all functions reachable from it. The
  /// calculation is very naive since it does not take into account loops, so
  /// if a function is called within a loop several times (where the trip count
  /// is unknown), we assume that it will be called just the once. The same
  /// applies if any of the callees are recursive (the function itself is
  /// unlikely to be recursive since it will have been outlined from a tapir
  /// loop).
  KernelInstMixData run(const Function &f) {
    KernelInstMixData res;
    // The set of reachable functions will contain the given function.
    for (const Function *f : reachable(f))
      res += calculate(*f);
    return res;
  }
};

static TapirTargetID getTapirTargetArg(CallBase &call) {
  return TapirTargetID(
      cast<ConstantInt>(call.getArgOperand(0))->getZExtValue());
}

static StringRef getKernelNameArg(CallBase &call) {
  // The first string that is passed to the call will be the kernel name. It is
  // reasonable to expect that this will never change since there is little
  // reason to use a string to represent the tapir target id.
  for (Use &op : call.args())
    if (auto *g = dyn_cast<GlobalVariable>(&*op))
      if (g->hasInitializer())
        if (auto *cda = dyn_cast<ConstantDataArray>(g->getInitializer()))
          if (cda->isCString())
            return cda->getAsCString();
  llvm_unreachable("Could not find kernel name argument in kernel launch call");
}

static GlobalVariable *getKernelMetadataArg(CallBase &call) {
  for (Use &op : call.args())
    if (auto *g = dyn_cast<GlobalVariable>(&*op))
      if (hasKitsuneKernelMDMD(*g))
        return g;
  return nullptr;
}

namespace llvm {

PreservedAnalyses FinalizeKernelMetadataPass::run(Module &m,
                                                  ModuleAnalysisManager &mam) {
  // If no primary tapir target has been set, the tapir target options will
  // not have been set, so there is nothing that we can do.
  const TapirTargetInfo &tgi = mam.getResult<TapirTargetAnalysis>(m);
  if (not tgi.hasID())
    return PreservedAnalyses::all();

  LLVMContext &ctx = m.getContext();
  Type *i64 = Type::getInt64Ty(ctx);
  StructType *kernelInstMixTy = getKernelInstMixType(ctx);
  EmbeddedModulesMapTy embMs = getEmbeddedModules(m);
  Function *launchFn =
      Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kitrt_launch_kernel);

  // FIXME: Currently, HipABI does not correctly create a global variable for
  // the kernel metadata, so don't do anything. When HiPABI is fixed, this
  // should go away.
  if (embMs.size() == 0)
    return PreservedAnalyses::all();

  InstMixCalculator instMixCalc;
  for (Use &u : launchFn->uses()) {
    if (auto *call = dyn_cast<CallBase>(u.getUser())) {
      if (call->getCalledFunction() == launchFn) {
        TapirTargetID tt = getTapirTargetArg(*call);
        StringRef kname = getKernelNameArg(*call);
        GlobalVariable *g = getKernelMetadataArg(*call);

        assert(g && "Missing argument containing kernel metadata");
        assert(embMs.find(tt) != embMs.end() &&
               "Embedded module for tapir target not found");

        Function *kf = embMs.at(tt)->getFunction(kname);
        assert(kf && "Could not find kernel function being launched");

        KernelInstMixData instMix = instMixCalc.run(*kf);
        LLVM_DEBUG(dbgs() << "\tinstruction mix:\n"
                          << "\t  " << kname << "\n"
                          << "\t    memory ops:  " << instMix.memOps << "\n"
                          << "\t    fp ops:      " << instMix.fpOps << "\n"
                          << "\t    integer ops: " << instMix.intOps << "\n"
                          << "\t    other ops:   " << instMix.otherOps << "\n");

        g->setInitializer(ConstantStruct::get(
            kernelInstMixTy, ConstantInt::get(i64, instMix.memOps),
            ConstantInt::get(i64, instMix.fpOps),
            ConstantInt::get(i64, instMix.intOps),
            ConstantInt::get(i64, instMix.otherOps)));
      }
    }
  }

  // At best, the initializers of one or more globals will have changed, but
  // nothing else, so all analyses on the module remain valid.
  return PreservedAnalyses::all();
}

} // namespace llvm
