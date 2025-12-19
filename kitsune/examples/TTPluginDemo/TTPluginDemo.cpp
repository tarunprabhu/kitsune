//===- TTPluginDemo.cpp - Example tapir target plugin ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple tapir target plugin to demonstrate how one may be built. This is a
// very silly example that simply adds a print statement to the body of every
// tapir loop.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TTPlugin.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

using namespace llvm;

class BookendLOP : public LoopOutlineProcessor {
public:
  BookendLOP(Module &m, const TTOptions &tto)
      : LoopOutlineProcessor(m, m, tto,
                             CloneFunctionChangeType::GlobalChanges) {}
  ~BookendLOP() {};

  /// Processes a call to an outlined helper function for a tapir loop \ref tl.
  void processOutlinedLoopCall(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                               DominatorTree &dt) override final {
    LLVMContext &ctx = M.getContext();
    Type *voidTy = Type::getVoidTy(ctx);
    FunctionType *bookendTy = FunctionType::get(voidTy, {}, /*VarArgs=*/false);
    FunctionCallee bookend = M.getOrInsertFunction("bookend", bookendTy);

    CallBase *call = cast<CallBase>(toi.ReplCall);
    CallInst::Create(bookend, {}, "")->insertAfter(call->getIterator());
    CallInst::Create(bookend, {}, "")->insertBefore(call->getIterator());
  }
};

/// Tapir target that adds calls to functions immediately before and after a
/// tapir loop
class BookendTT : public TapirTarget {
public:
  BookendTT(Module &m, const TTOptions &tto) : TapirTarget(m, tto) {};
  virtual ~BookendTT() = default;

  void lowerSync(SyncInst &) override final {}
  void postProcessFunction(Function &, bool) override final {}
  void postProcessHelper(Function &) override final {}
  void preProcessOutlinedTask(Function &, Instruction *, Instruction *, bool,
                              BasicBlock *) override final {}
  void postProcessOutlinedTask(Function &, Instruction *, Instruction *, bool,
                               BasicBlock *) override final {}
  void preProcessRootSpawner(Function &, BasicBlock *) override final {}
  void postProcessRootSpawner(Function &, BasicBlock *) override final {}
  void processSubTaskCall(TaskOutlineInfo &, DominatorTree &) override final {}

  bool shouldDoOutlining(const Function &) const override final { return true; }

  // Replace uses of grainsize intrinsic call with this grainsize value.
  Value *lowerGrainsizeCall(CallInst *call) override final {
    Value *grainSize = ConstantInt::get(call->getType(), 1);
    call->replaceAllUsesWith(grainSize);
    return grainSize;
  }

  bool preProcessFunction(Function &, TaskInfo &, bool) override final {
    // We don't do anything
    return false;
  };

  LoopOutlineProcessor *
  getLoopOutlineProcessor(const TapirLoopInfo *tl) override final {
    return new BookendLOP(M, this->getOptions());
  }
};

// This externally visible function with C linkage is required. It is the
// well-known entry point required by LLVM.
//
// The compiler and linker options that are returned have been chosen to be
// relatively "innocuous". This plugin is used in Kitsune's core tests, so we
// do need something that is unlikely to unexpectedly change the compiler's
// output.
extern "C" ::llvm::TTPluginInfo LLVM_ATTRIBUTE_WEAK llvmGetTTPluginInfo() {
  return {LLVM_TTPLUGIN_API_VERSION,
          "TTPluginDemo",
          "1.0",
          [](Module &hostM, const TTOptions &tto) -> TapirTarget * {
            return new BookendTT(hostM, tto);
          },
          []() -> TTPlugin::ExtraArgsList {
            return {"-O"};
          },
          []() -> TTPlugin::ExtraArgsList {
            return {"-L/path/to/something/that/does/not/exist"};
          }};
}
