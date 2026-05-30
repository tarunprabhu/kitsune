//===- GenerateCtors.cpp - Generate global ctors and dtors ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates global constructors and destructors needed by Kitsune.
//
// These will initialize and finalize kitsune's runtime(s). They will do the
// same for any other runtimes (such as cuda and hip). These may involve
// registering global variables and fat binaries with the underlying
// GPU-specific runtime, setting environment variables etc. Not all tapir
// targets require Kitsune's runtime, but this pass will always be run when
// tapir is enabled.
//
// In addition to creating the constructors and destructors, this pass will
// also create any any global variables needed by the global ctor. In the case
// of the GPU tapir targets and associated runtimes, these include globals for
// the fat binary, the bundle that wraps the fat binary etc.
//
// This pass should only be run once per module and should be run as late as
// possible to ensure that all tapir targets have been run already.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/GenerateCtors.h"
#include "GenerateCtorsImpl.h"
#include "kitsune/Analysis/TTObjectsAnalysis.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

// FIXME: We really should not be exposing command line options from other
// source files. These are experimental options that have been hacked in for the
// moment. If this is useful, we should consider adding it to the tapir target
// options instead. Otherwise, it should be removed altogether.
//
// These are declared [[weak]] because they are defined in one of the tapir
// targets. The tapir targets are not guaranteed to be built, therefore, these
// globals may not be available at link time.
//
#ifdef __APPLE__
#define WEAK weak_import
#else
#define WEAK weak
#endif
extern __attribute__((WEAK)) cl::opt<bool> clRefineLaunches;
extern __attribute__((WEAK)) cl::opt<bool> clUseYLaunch;

/// Is the given intrinsic \param id called at least once in the module \param m
/// with the tapir target id \param tt
static bool isCalledWithTTID(Module &m, Intrinsic::ID id, TTID tt) {
  if (Function *f = m.getFunction(Intrinsic::getBaseName(id)))
    for (Use &u : f->uses())
      if (auto *call = dyn_cast<CallBase>(u.getUser()))
        // Although unlikely, the intrinsic could have been passed as an
        // argument to some other function. Just in case, check that the callee
        // at this site is the launch kernel function.
        if (call->getCalledFunction() == f)
          if (auto *cint = dyn_cast<ConstantInt>(call->getArgOperand(0)))
            if (std::optional<TTID> ttid = fromConstant<TTID>(*cint))
              if (*ttid == tt)
                return true;
  return false;
}

/// Check if any functions from Cilk's runtime are used.
static bool usesCilkRT(Module &m) {
  // __cilkrts_status is probably safe to check for since it is likely to be
  // used in most cases. Checking for a cilk stack frame is better, but the
  // stack frame handling functions get inlined making it difficult to be
  // certain that it is used. We could check that a stack frame is alloca'ed,
  // but that would involve checking the type name. This is not reliable,
  // especially on large codes at high optimization levels because the struct
  // type may be mangled, or replaced with an anonymous type.
  if (m.getGlobalVariable("__cilkrts_status"))
    return true;

  // There is a chance that the function will not have been inlined, so we may
  // as well check for that in case __cilkrts_status is not present.
  if (m.getFunction("__cilkrts_enter_frame"))
    return true;

  // The two cases above should handle the common case. But in case they fail,
  // attempt to find the creation of a cilk stack frame. The stack frame will
  // have been created in the entry block of the function, so just look there.
  //
  // Ideally, we should only look in the outlined function, but there is no way
  // to reliably identify such functions.
  for (Function &f : m.functions())
    if (f.size())
      for (Instruction &inst : f.getEntryBlock())
        if (auto *alloca = dyn_cast<AllocaInst>(&inst))
          if (auto *sty = dyn_cast<StructType>(alloca->getAllocatedType()))
            if (sty->hasName())
              return sty->getName().starts_with("__cilkrts_stack_frame");

  return false;
}

/// Should a ctor be generated for a tapir target.
static bool shouldGenerateCtor(Module &m, TTID tt) {
  switch (tt) {
  case TTID::Cuda:
  case TTID::Hip:
    return isCalledWithTTID(m, Intrinsic::kit_async_gpu_kernel_launch, tt);
  case TTID::OpenCilk:
    return usesCilkRT(m);
  case TTID::OpenMP:
    return isCalledWithTTID(m, Intrinsic::kit_cpu_threads_launch, tt);
  case TTID::Pthreads:
    return isCalledWithTTID(m, Intrinsic::kit_async_cpu_threads_launch, tt);
  case TTID::Qthreads:
    return isCalledWithTTID(m, Intrinsic::kit_cpu_threads_launch, tt);
  default:
    llvm_unreachable("shouldGenereateCtor: TTID not handled");
  }
}

static const std::map<TTID, detail::GenerateCtorImplFn> genCtorFns = {
    {TTID::Cuda, detail::genCtorCuda},
    {TTID::Hip, detail::genCtorHip},
    {TTID::OpenCilk, detail::genCtorOpenCilk},
    {TTID::OpenMP, detail::genCtorOpenMP},
    {TTID::Pthreads, detail::genCtorPthreads},
    {TTID::Qthreads, detail::genCtorQthreads},
};

PreservedAnalyses GenerateCtorsPass::run(Module &m,
                                         ModuleAnalysisManager &mam) {
  // If no primary tapir target has been set, there will be nothing to do, so
  // bail out immediately.
  const TTObjects &ttObjs = mam.getResult<TTObjectsAnalysis>(m);
  if (not ttObjs.hasTTID())
    return PreservedAnalyses::all();

  auto &fam = mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();
  auto getTLI = [&](Function &f) -> TargetLibraryInfo & {
    return fam.getResult<TargetLibraryAnalysis>(f);
  };

  detail::GenerateCtorOptions genCtorOpts;
  if (&clRefineLaunches)
    genCtorOpts.refineLaunches = clRefineLaunches;
  if (&clUseYLaunch)
    genCtorOpts.useYLaunch = clUseYLaunch;

  const TTOptions &ttOpts = ttObjs.getOptions();
  for (const auto &[tt, genCtorFn] : genCtorFns)
    if (shouldGenerateCtor(m, tt))
      genCtorFn(m, getTLI, ttOpts, genCtorOpts);

  // This never invalidates any analyses since, at most, only the initializer of
  // a global variable will have changed. The generated ctors will not be called
  // explicitly in the code, so the callgraph will not have changed either.
  return PreservedAnalyses::all();
}
