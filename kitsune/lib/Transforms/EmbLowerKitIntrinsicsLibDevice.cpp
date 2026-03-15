//=- EmbLowerKitIntrinsicsLibDevice.cpp - Lower Kitsune-specific intrinsics -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower those Kitsune-specific intrinsics in an embedded module that must be
// lowered to functions provided by a libdevice bitcode file.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EmbLowerKitIntrinsicsLibDevice.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"

#include <map>

#define DEBUG_TYPE "emb-lower-intrinsics-libdevice"

using namespace llvm;

static bool lowerKitIntrinsicsCuda(Module &embM) {
  // There are currently no Kitsune-specific intrinsics that must be lowered
  // to libdevice functions.
  return false;
}

static bool lowerHipBlockGridSizeIntrinsics(Module &embM) {
  constexpr StringRef getLocalSize = "__ockl_get_local_size";
  constexpr StringRef getGlobalSize = "__ockl_get_global_size";

  std::map<CallBase *, std::tuple<StringRef, unsigned>> calls;
  for (Function &f : embM)
    for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
      if (auto *call = dyn_cast<CallBase>(&*i))
        if (Intrinsic::ID callee = call->getIntrinsicID()) {
          if (callee == Intrinsic::kit_gpu_block_size_x)
            calls.insert({call, {getLocalSize, 0}});
          else if (callee == Intrinsic::kit_gpu_block_size_y)
            calls.insert({call, {getLocalSize, 1}});
          else if (callee == Intrinsic::kit_gpu_block_size_z)
            calls.insert({call, {getLocalSize, 2}});
          else if (callee == Intrinsic::kit_gpu_grid_size_x)
            calls.insert({call, {getGlobalSize, 0}});
          else if (callee == Intrinsic::kit_gpu_grid_size_y)
            calls.insert({call, {getGlobalSize, 1}});
          else if (callee == Intrinsic::kit_gpu_grid_size_z)
            calls.insert({call, {getGlobalSize, 2}});
        }

  LLVMContext &ctx = embM.getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);

  for (const auto &p : calls) {
    CallBase *call = p.first;
    StringRef libf = std::get<StringRef>(p.second);
    unsigned dirxn = std::get<unsigned>(p.second);

    StringRef name = call->getName();
    Type *type = call->getType();
    FunctionCallee newCallee = embM.getOrInsertFunction(libf, i64, i32);
    Constant *farg = ConstantInt::get(i32, dirxn, /*isSigned=*/false);

    IRBuilder<> builder(call);
    Value *newCall = builder.CreateCall(newCallee, {farg}, name + ".64");
    Value *newVal = builder.CreateIntCast(newCall, type, /*isSigned=*/false);

    // Set the name after removing the original call. Doing so earlier may
    // result in it being renamed to avoid a conflict with the name of the call
    // instruction being replaced.
    call->replaceAllUsesWith(newVal);
    call->eraseFromParent();
    newVal->setName(name);
  }

  return calls.size();
}

static bool lowerKitIntrinsicsHip(Module &embM) {
  bool changed = false;

  changed |= lowerHipBlockGridSizeIntrinsics(embM);

  return changed;
}

static bool lowerKitIntrinsics(TTID tt, Module &embM) {
  switch (tt) {
  case TTID::Cuda:
    return lowerKitIntrinsicsCuda(embM);
  case TTID::Hip:
    return lowerKitIntrinsicsHip(embM);
  default:
    break;
  }
  llvm_unreachable("lowerKitIntrinsics: TTID not handled");
}

bool EmbLowerKitIntrinsicsLibDevicePass::run(TTID tt, Module &devM,
                                             Module &hostM,
                                             ModuleAnalysisManager &hostMAM) {
  return lowerKitIntrinsics(tt, devM);
}
