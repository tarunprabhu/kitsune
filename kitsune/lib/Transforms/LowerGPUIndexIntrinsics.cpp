//===- LowerGPUIndexIntrinsics.cpp - Lower GPU index intrinsics -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune-specific GPU index intrinsics.
//
//===----------------------------------------------------------------------===//

#include "LowerGPUIntrinsicsImpl.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/IntrinsicUtils.h"
#include "kitsune/Core/ModuleUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

static bool replaceSimple(CallBase *call, Intrinsic::ID newIntr) {
  // The first argument of the call will be the TTID. This is never needed when
  // lowering a simple call.
  SmallVector<Value *, 4> args;
  for (unsigned i = 1; i < call->arg_size(); ++i)
    args.push_back(call->getArgOperand(i));

  Module *m = call->getModule();
  Function *f = Intrinsic::getOrInsertDeclaration(m, newIntr);
  CallInst *newCall = CallInst::Create(f, args);
  ReplaceInstWithInst(call, newCall);

  return true;
}

// Replace a call with another to a function with the given name. A
// declaration for the function is added since it is assumed that the
// definition will become available at link-time. The function takes a single
// integer argument, which is also provided.
static bool replaceLibDevice(CallInst *call, StringRef fname, unsigned dirxn) {
  Module *m = call->getModule();
  LLVMContext &ctx = m->getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);

  Type *type = call->getType();
  Function *libF = getOrInsertFunction(*m, fname, i64, i32);
  Constant *farg = toConstant(dirxn, ctx);

  CallInst *newCall = CallInst::Create(libF, {farg});
  if (libF->getReturnType() != type) {
    CastInst *newVal =
        CastInst::CreateIntegerCast(newCall, type, /*isSigned=*/false);
    ReplaceInstWithInst(call, newVal);
    newCall->insertBefore(newVal->getIterator());
  } else {
    ReplaceInstWithInst(call, newCall);
  }
  return true;
};

static bool lowerForHip(CallInst *call) {
  // These functions are provide by ockl.bc - a bitcode file that is part of
  // the ROCm installation.
  constexpr StringRef getLocalSize = "__ockl_get_local_size";
  constexpr StringRef getGlobalSize = "__ockl_get_global_size";

  switch (call->getIntrinsicID()) {
  case Intrinsic::kit_gpu_thread_id_x:
    return replaceSimple(call, Intrinsic::amdgcn_workitem_id_x);
  case Intrinsic::kit_gpu_thread_id_y:
    return replaceSimple(call, Intrinsic::amdgcn_workitem_id_y);
  case Intrinsic::kit_gpu_thread_id_z:
    return replaceSimple(call, Intrinsic::amdgcn_workitem_id_z);
  case Intrinsic::kit_gpu_block_id_x:
    return replaceSimple(call, Intrinsic::amdgcn_workgroup_id_x);
  case Intrinsic::kit_gpu_block_id_y:
    return replaceSimple(call, Intrinsic::amdgcn_workgroup_id_y);
  case Intrinsic::kit_gpu_block_id_z:
    return replaceSimple(call, Intrinsic::amdgcn_workgroup_id_z);
  case Intrinsic::kit_gpu_block_size_x:
    return replaceLibDevice(call, getLocalSize, 0);
  case Intrinsic::kit_gpu_block_size_y:
    return replaceLibDevice(call, getLocalSize, 1);
  case Intrinsic::kit_gpu_block_size_z:
    return replaceLibDevice(call, getLocalSize, 2);
  case Intrinsic::kit_gpu_grid_size_x:
    return replaceLibDevice(call, getGlobalSize, 0);
  case Intrinsic::kit_gpu_grid_size_y:
    return replaceLibDevice(call, getGlobalSize, 1);
  case Intrinsic::kit_gpu_grid_size_z:
    return replaceLibDevice(call, getGlobalSize, 2);
  default: break;
  }
  llvm_unreachable("replaceKitIntrinsicsEarlyHip: Unexpected intrinsic");
}

static bool lowerForCuda(CallInst *call) {
  switch (call->getIntrinsicID()) {
  case Intrinsic::kit_gpu_thread_id_x:
    return replaceSimple(call, Intrinsic::nvvm_read_ptx_sreg_tid_x);
  case Intrinsic::kit_gpu_thread_id_y:
    return replaceSimple(call, Intrinsic::nvvm_read_ptx_sreg_tid_y);
  case Intrinsic::kit_gpu_thread_id_z:
    return replaceSimple(call, Intrinsic::nvvm_read_ptx_sreg_tid_z);
  case Intrinsic::kit_gpu_block_id_x:
    return replaceSimple(call, Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  case Intrinsic::kit_gpu_block_id_y:
    return replaceSimple(call, Intrinsic::nvvm_read_ptx_sreg_ctaid_y);
  case Intrinsic::kit_gpu_block_id_z:
    return replaceSimple(call, Intrinsic::nvvm_read_ptx_sreg_ctaid_z);
  case Intrinsic::kit_gpu_block_size_x:
    return replaceSimple(call, Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  case Intrinsic::kit_gpu_block_size_y:
    return replaceSimple(call, Intrinsic::nvvm_read_ptx_sreg_ntid_y);
  case Intrinsic::kit_gpu_block_size_z:
    return replaceSimple(call, Intrinsic::nvvm_read_ptx_sreg_ntid_z);
  case Intrinsic::kit_gpu_grid_size_x:
    return replaceSimple(call, Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
  case Intrinsic::kit_gpu_grid_size_y:
    return replaceSimple(call, Intrinsic::nvvm_read_ptx_sreg_nctaid_y);
  case Intrinsic::kit_gpu_grid_size_z:
    return replaceSimple(call, Intrinsic::nvvm_read_ptx_sreg_nctaid_z);
  default: break;
  }
  llvm_unreachable("lowerGPUIndexIntrinsic[cuda]: Unexpected intrinsic");
}

bool llvm::detail::lowerGPUIndexIntr(CallInst *call) {
  switch (*getTTIDFromKitIntrCall(*call)) {
  case TTID::Cuda: return lowerForCuda(call);
  case TTID::Hip: return lowerForHip(call);
  default: break;
  }
  llvm_unreachable("lowerGPUIndexIntrinsic: TTID not handled");
}
