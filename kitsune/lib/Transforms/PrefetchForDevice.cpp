//===- PrefetchForDevice.cpp - Generate dtoh/htod prefetch calls ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass generates calls to initiate movement of data between host and
// device. This will only generate calls to Kitsune's prefetch intrinsics. This
// is typically run early in Kitsune's post-tapir pipeline, but it may be
// run later in the pipeline as well. This should only modify the host,
// but it may be profitable to examine the embedded device modules when
// deciding if/when to prefetch.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/PrefetchForDevice.h"
#include "kitsune/Core/AddrSpace.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/IntrinsicUtils.h"
#include "kitsune/Core/TTOptions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "kit-prefetch"

using namespace llvm;

namespace {

/// Implementation of the prefetch pass. This is currently very simple and
/// inserts device-to-host prefetches immediately before kernel launches as long
/// as prefetching has not been disabled in the tapir target options. This
/// implementation does not distinguish between launches on NVIDIA or AMD GPU's.
/// This is unlikely to be profitable in the general case. This pass will almost
/// certainly have to be split into special prefetching implementations for
/// NVIDIA and AMD and will probably need to query the embedded device modules
/// to determine exactly which GPU to generate prefetches for do something
/// different for each of them. If that happens, this comment should be updated
/// to reflect any changes.
class Prefetch {
private:
  bool insertPrefetchesHToD(CallBase &call);
  bool insertPrefetchesDToH(CallBase &call);
  bool insertPrefetches(CallBase &call);

public:
  bool run(Module &m);
};

} // namespace

/// Insert prefetches from host to device before the given kernel launch call.
/// Currently, this inserts the prefetches immediately before the call. It may
/// be beneficial to insert these earlier in the function. That optimization
/// may be implemented at some point.
bool Prefetch::insertPrefetchesHToD(CallBase &call) {
  LLVMContext &ctx = call.getContext();
  PointerType *ptrTy = PointerType::getUnqual(ctx);

  bool changed = false;
  Value *stream = getStreamFromLaunch(call);
  for (Value *arg : getVariadicArgs(call)) {
    if (auto *pty = dyn_cast<PointerType>(arg->getType())) {
      IRBuilder<> builder(&call);

      // These should be required to be pointers in Kitsune's mobile address
      // space. Currently, the frontend support for this is a bit spotty, so
      // Kitsune's intrinsics operate on pointers in the default address space.
      // But this really should change at some point.
      //
      // For now, if the pointer is not in the default address space, make
      // sure that it is.
      if (pty->getAddressSpace() != KitAS::Default)
        arg = builder.CreateAddrSpaceCast(arg, ptrTy);

      // The pointer to the data to be prefetched must point to UVM allocated
      // memory. By setting the number of bytes to be prefetched to -1, we are
      // instructing the runtime to prefetch the entire UVM-allocated buffer.
      // The runtime keeps track of this.
      //
      // TODO: Do some analysis to only prefetch the number of bytes that are
      // actually used (or likely to be used) by the kernel.
      Constant *bytes = toConstant(-1L, ctx);

      Value *ctt = call.getArgOperand(0);
      builder.CreateIntrinsic(Intrinsic::kit_async_gpu_prefetch_htod,
                              {ctt, arg, bytes, stream});

      changed |= true;
    }
  }

  return changed;
}

/// Insert prefetches from device to host after the given kernel launch call.
bool Prefetch::insertPrefetchesDToH(CallBase &call) {
  // TODO: Implement this.
  return false;
}

/// Insert prefetches for the given kernel launch call if necessary. Return
/// true if any prefetch calls were inserted, false otherwise.
bool Prefetch::insertPrefetches(CallBase &call) {
  bool changed = false;

  changed |= insertPrefetchesHToD(call);
  changed |= insertPrefetchesDToH(call);

  return changed;
}

/// Return true if at least one prefetch call was added to a function in the
/// module \p m, false otherwise.
bool Prefetch::run(Module &m) {
  bool changed = false;
  if (Function *launch = Intrinsic::getDeclarationIfExists(
          &m, Intrinsic::kit_async_gpu_kernel_launch))
    for (Use &u : launch->uses())
      if (auto *call = dyn_cast<CallBase>(u.getUser()))
        if (call->getCalledFunction() == launch)
          changed |= insertPrefetches(*call);
  return changed;
}

PreservedAnalyses PrefetchForDevicePass::run(Module &m,
                                             ModuleAnalysisManager &am) {
  // If no primary tapir target has been set, there will be nothing to do, so
  // bail out immediately.
  if (!tto.hasTTID() || !tto.getGPUPrefetch())
    return PreservedAnalyses::all();

  if (Prefetch().run(m))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
