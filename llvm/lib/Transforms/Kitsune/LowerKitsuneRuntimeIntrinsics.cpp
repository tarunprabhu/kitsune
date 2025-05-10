//==- LowerKitsuneRuntimeIntrinsics.cpp - Lower kitrt intrinsics -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers intrinsics that correspond to functions in Kitsune's
// runtime.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Kitsune/LowerKitsuneRuntimeIntrinsics.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Frontend/Tapir/Tapir.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"

#include <map>

using namespace llvm;

#define DEBUG_TYPE "lower-kitsune-runtime-intrinsics"

namespace {

using KitsuneRuntimeFuncMap = std::map<Intrinsic::ID, LibFunc>;

/// Kitsune runtime functions for any tapir target.
static const KitsuneRuntimeFuncMap kitFuncs = {
    {Intrinsic::kitrt_enable_verbose, LibFunc_kitrt_enable_verbose},
};

/// Kitsune runtime functions for the cuda tapir target.
static const KitsuneRuntimeFuncMap kitCudaFuncs = {
    {Intrinsic::kitrt_enable_refine_launches,
     LibFunc_kitcuda_enable_refine_launches},
    {Intrinsic::kitrt_finalize, LibFunc_kitcuda_finalize},
    {Intrinsic::kitrt_initialize, LibFunc_kitcuda_initialize},
    {Intrinsic::kitrt_launch_kernel, LibFunc_kitcuda_launch_kernel},
    {Intrinsic::kitrt_prefetch_device, LibFunc_kitcuda_prefetch_device},
    {Intrinsic::kitrt_prefetch_host, LibFunc_kitcuda_prefetch_host},
    {Intrinsic::kitrt_set_fixed_tpb, LibFunc_kitcuda_set_fixed_tpb},
    {Intrinsic::kitrt_set_max_tpb, LibFunc_kitcuda_set_max_tpb},
    {Intrinsic::kitrt_symbol_device_ptr, LibFunc_kitcuda_symbol_device_ptr},
    {Intrinsic::kitrt_symbol_memcpy_device,
     LibFunc_kitcuda_symbol_memcpy_device},
    {Intrinsic::kitrt_symbol_memcpy_host, LibFunc_kitcuda_symbol_memcpy_host},
    {Intrinsic::kitrt_sync_stream, LibFunc_kitcuda_sync_stream},
};

/// Kitsune runtime functions for the hip tapir target.
static const KitsuneRuntimeFuncMap kitHipFuncs = {
    {Intrinsic::kitrt_enable_y_axis_launches,
     LibFunc_kithip_enable_y_axis_launches},
    {Intrinsic::kitrt_finalize, LibFunc_kithip_finalize},
    {Intrinsic::kitrt_hip_enable_xnack, LibFunc_kithip_enable_xnack},
    {Intrinsic::kitrt_initialize, LibFunc_kithip_initialize},
    {Intrinsic::kitrt_launch_kernel, LibFunc_kithip_launch_kernel},
    {Intrinsic::kitrt_prefetch_device, LibFunc_kithip_prefetch_device},
    {Intrinsic::kitrt_prefetch_host, LibFunc_kithip_prefetch_host},
    {Intrinsic::kitrt_set_fixed_tpb, LibFunc_kithip_set_fixed_tpb},
    {Intrinsic::kitrt_set_max_tpb, LibFunc_kithip_set_max_tpb},
    {Intrinsic::kitrt_symbol_device_ptr, LibFunc_kithip_symbol_device_ptr},
    {Intrinsic::kitrt_symbol_memcpy_device,
     LibFunc_kithip_symbol_memcpy_device},
    {Intrinsic::kitrt_symbol_memcpy_host, LibFunc_kithip_symbol_memcpy_host},
    {Intrinsic::kitrt_sync_stream, LibFunc_kithip_sync_stream},
};

/// Runtime library function maps for tapir targets that have a corresponding
/// kitsune runtime.
static const std::map<TapirTargetID, KitsuneRuntimeFuncMap> kitTTFuncs = {
    {TapirTargetID::Cuda, kitCudaFuncs},
    {TapirTargetID::Hip, kitHipFuncs},
};

class LowerKitsuneRuntimeIntrinsicsImpl {
private:
  TargetLibraryInfo &tli;

private:
  FunctionCallee getOrInsertLibFunc(Module &m, LibFunc libFunc,
                                    FunctionType *fty) {
    FunctionCallee f = llvm::getOrInsertLibFunc(&m, tli, libFunc, fty);
    inferNonMandatoryLibFuncAttrs(*cast<Function>(f.getCallee()), tli);
    return f;
  }

  /// Check if the given id is a kitsune runtime intrinsic.
  bool isKitsuneRuntimeIntrinsic(Intrinsic::ID id) const {
    switch (id) {
    case Intrinsic::kitrt_enable_verbose:
    case Intrinsic::kitrt_enable_refine_launches:
    case Intrinsic::kitrt_enable_y_axis_launches:
    case Intrinsic::kitrt_hip_enable_xnack:
    case Intrinsic::kitrt_finalize:
    case Intrinsic::kitrt_initialize:
    case Intrinsic::kitrt_launch_kernel:
    case Intrinsic::kitrt_prefetch_device:
    case Intrinsic::kitrt_prefetch_host:
    case Intrinsic::kitrt_set_fixed_tpb:
    case Intrinsic::kitrt_set_max_tpb:
    case Intrinsic::kitrt_symbol_device_ptr:
    case Intrinsic::kitrt_symbol_memcpy_device:
    case Intrinsic::kitrt_symbol_memcpy_host:
    case Intrinsic::kitrt_sync_stream:
      return true;
    default:
      break;
    }
    assert(!Intrinsic::getBaseName(id).starts_with("llvm.kitrt") &&
           "Intrinsic may be a kitsune runtime intrinsic but not recognized");
    return false;
  }

  /// Some runtime intrinsics take the tapir target id as the first argument.
  /// Get the tapir target from this argument. It is an error to call this
  /// function with a call that is not a kitsune runtime intrinsic and does not
  /// have a valid tapir target as the first argument.
  TapirTargetID getTapirTargetID(CallBase &call) {
    if (auto *cint = dyn_cast<ConstantInt>(call.getArgOperand(0)))
      return TapirTargetID(cint->getZExtValue());
    llvm_unreachable("getTapirTargetID: Not a valid tapir target id");
  }

  /// Get the signature of the kitsune runtime function that corresponds to the
  /// given kitrt intrinsic.
  FunctionType *getKitsuneRuntimeFuncSig(CallBase &call) {
    std::vector<Type *> params;
    FunctionType *fty = call.getFunctionType();
    LLVMContext &ctxt = call.getContext();

    switch (call.getIntrinsicID()) {

      // This takes a single boolean argument. In the intrinsic, the boolean is
      // of type i1, but in the runtime functions, booleans are always i8.
    case Intrinsic::kitrt_enable_refine_launches:
      params = {IntegerType::get(ctxt, 8)};
      break;

      // These intrinsics take the tapir target id as the first argument. This
      // is not passed to the corresponding kitsune runtime function. The other
      // arguments are passed as is.
    case Intrinsic::kitrt_finalize:
    case Intrinsic::kitrt_initialize:
    case Intrinsic::kitrt_launch_kernel:
    case Intrinsic::kitrt_set_fixed_tpb:
    case Intrinsic::kitrt_set_max_tpb:
    case Intrinsic::kitrt_symbol_device_ptr:
    case Intrinsic::kitrt_symbol_memcpy_device:
    case Intrinsic::kitrt_symbol_memcpy_host:
    case Intrinsic::kitrt_sync_stream:
      for (unsigned i = 1; i < fty->getNumParams(); ++i)
        params.push_back(fty->getParamType(i));
      break;

      // The kitsune prefetch intrinsics take the number of bytes to prefetch
      // as an argument. Currently, the corresponding runtime functions do not
      // and will prefetch all of the data allocated by the UVM-allocated
      // buffer being prefetched.
    case Intrinsic::kitrt_prefetch_device:
    case Intrinsic::kitrt_prefetch_host:
      params = {fty->getParamType(1), fty->getParamType(3)};
      break;

      // The intrinsics take a boolean argument, but the corresponding runtime
      // functions do not.
    case Intrinsic::kitrt_enable_verbose:
    case Intrinsic::kitrt_enable_y_axis_launches:
    case Intrinsic::kitrt_hip_enable_xnack:
      break;

    default:
      llvm_unreachable("getKitsuneRuntimeFuncSig: IntrinsicID not handled");
    }
    return FunctionType::get(fty->getReturnType(), params, false);
  }

  /// Get the kitsune runtime function that will replace the intrinsic called in
  /// the given call instruction.
  FunctionCallee getKitsuneRuntimeFunc(CallBase &call) {
    Intrinsic::ID id = call.getIntrinsicID();
    Module *m = call.getModule();
    FunctionType *fty = getKitsuneRuntimeFuncSig(call);

    switch (id) {

      // Intrinsics with runtime functions that are independent of a tapir
      // target.
    case Intrinsic::kitrt_enable_verbose:
      assert(
          kitFuncs.find(id) != kitFuncs.end() &&
          "getKitsuneRuntimeFunc: No kitsune library function for intrinsic");
      return getOrInsertLibFunc(*m, kitFuncs.at(id), fty);

      // Intrinsics that are exclusive to the hip tapir target
    case Intrinsic::kitrt_hip_enable_xnack: {
      const KitsuneRuntimeFuncMap &funcs = kitTTFuncs.at(TapirTargetID::Hip);
      assert(funcs.find(id) != funcs.end() &&
             "getKitsuneRuntimeFunc: No kitsune library function for the hip "
             "tapir target");
      return getOrInsertLibFunc(*m, funcs.at(id), fty);
    }

      // Intrinsics with runtime functions dependent on the tapir target.
    case Intrinsic::kitrt_enable_refine_launches:
    case Intrinsic::kitrt_enable_y_axis_launches:
    case Intrinsic::kitrt_finalize:
    case Intrinsic::kitrt_initialize:
    case Intrinsic::kitrt_prefetch_device:
    case Intrinsic::kitrt_prefetch_host:
    case Intrinsic::kitrt_launch_kernel:
    case Intrinsic::kitrt_set_fixed_tpb:
    case Intrinsic::kitrt_set_max_tpb:
    case Intrinsic::kitrt_symbol_device_ptr:
    case Intrinsic::kitrt_symbol_memcpy_device:
    case Intrinsic::kitrt_symbol_memcpy_host:
    case Intrinsic::kitrt_sync_stream: {
      TapirTargetID tt = getTapirTargetID(call);
      assert(kitTTFuncs.find(tt) != kitTTFuncs.end() &&
             "getKitsuneRuntimeFunc: Invalid tapir target for intrinsic");

      const KitsuneRuntimeFuncMap &funcs = kitTTFuncs.at(tt);
      assert(funcs.find(id) != funcs.end() &&
             "getKitsuneRuntimeFunc: No kitsune library function for tapir "
             "target");

      return getOrInsertLibFunc(*m, funcs.at(id), fty);
    }

    default:
      llvm_unreachable("getKitsuneRuntimeFunc: Library function not found");
    }
  }

  /// Construct a call instruction to replace the given call. The callee of the
  /// given call must be a Kitsune intrinsic. The result will be a call to a
  /// kitsune runtime function. If the given call must be removed instead of
  /// being replaced, nullptr must be returned.
  CallInst *getKitsuneRuntimeCall(CallBase &call) {
    std::vector<Value *> args;
    User::op_iterator argBegin = call.arg_begin();
    User::op_iterator argEnd = call.arg_end();

    switch (call.getIntrinsicID()) {

      // The only argument is an immediate flag. If the flag is false, the
      // corresponding runtime function should not be called.
    case Intrinsic::kitrt_enable_verbose:
    case Intrinsic::kitrt_hip_enable_xnack:
      if (cast<ConstantInt>(call.getArgOperand(0))->isZero())
        return nullptr;
      break;

      // The first argument is the tapir target id. The second is a boolean
      // immediate flag. If the flag is false, the corresponding runtime
      // function should not be called.
    case Intrinsic::kitrt_enable_y_axis_launches:
      if (cast<ConstantInt>(call.getArgOperand(1))->isZero())
        return nullptr;
      break;

      // The first argument is the tapir target id. The second is the flag that
      // must be passed to the corresponding runtime function.
    case Intrinsic::kitrt_enable_refine_launches:
      args.push_back(call.getArgOperand(1));
      break;

      // The intrinsic takes the tapir target id as the first argument. That
      // should not be passed to the runtime function.
    case Intrinsic::kitrt_finalize:
    case Intrinsic::kitrt_initialize:
    case Intrinsic::kitrt_launch_kernel:
    case Intrinsic::kitrt_set_fixed_tpb:
    case Intrinsic::kitrt_set_max_tpb:
    case Intrinsic::kitrt_symbol_device_ptr:
    case Intrinsic::kitrt_symbol_memcpy_device:
    case Intrinsic::kitrt_symbol_memcpy_host:
    case Intrinsic::kitrt_sync_stream:
      for (User::op_iterator arg = ++argBegin; arg != argEnd; ++arg)
        args.push_back(*arg);
      break;

      // Currently, the number of bytes to prefetch is passed as an argument to
      // the intrinsic, but the corresponding runtime function does not accept
      // this.
    case Intrinsic::kitrt_prefetch_device:
    case Intrinsic::kitrt_prefetch_host:
      args = {call.getArgOperand(1), call.getArgOperand(3)};
      break;

    default:
      llvm_unreachable("getKitsuneRuntimeArgs: IntrinsicID not handled");
    }

    FunctionCallee func = getKitsuneRuntimeFunc(call);
    return CallInst::Create(func, args, call.getName(), call.getIterator());
  }

public:
  LowerKitsuneRuntimeIntrinsicsImpl(TargetLibraryInfo &tli) : tli(tli) {}

  bool run(Function &f) {
    // The keys in this map are the call instructions to be replaced. If the
    // value is nullptr, the instruction is removed, otherwise, all uses of the
    // key are replaced with the value.
    std::map<CallBase *, CallBase *> repls;
    for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i) {
      if (auto *call = dyn_cast<CallBase>(&*i)) {
        if (isKitsuneRuntimeIntrinsic(call->getIntrinsicID())) {
          CallInst *repl = getKitsuneRuntimeCall(*call);
          repls[call] = repl;
        }
      }
    }
    for (auto &[call, repl] : repls) {
      if (repl)
        call->replaceAllUsesWith(repl);
      call->eraseFromParent();
    }
    return repls.size();
  }
};

} // namespace

namespace llvm {

PreservedAnalyses
LowerKitsuneRuntimeIntrinsicsPass::run(Module &m, ModuleAnalysisManager &mam) {
  bool changed = false;
  auto &fam = mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();

  for (Function &f : m) {
    TargetLibraryInfo &tli = fam.getResult<TargetLibraryAnalysis>(f);

    changed |= LowerKitsuneRuntimeIntrinsicsImpl(tli).run(f);
  }

  // If any kitsune intrinsics were replaced, the call graph will have changed,
  // but other analyses will not have been invalidated.
  if (changed) {
    PreservedAnalyses pa;
    pa.preserve<FunctionAnalysisManagerCGSCCProxy>();
    pa.preserveSet<AllAnalysesOn<Function>>();
    return pa;
  }
  return PreservedAnalyses::all();
}

} // namespace llvm
