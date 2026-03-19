//===- TapirTargets.cpp - Utilities for tapir target objects --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to create tapir target objects, query enabled tapir targets etc.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Targets/TapirTargets.h"
#include "kitsune/Config/Config.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Support/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"

#if KITSUNE_CUDA_ENABLED
#include "kitsune/Targets/CudaABI.h"
#endif

#if KITSUNE_HIP_ENABLED
#include "kitsune/Targets/HipABI.h"
#endif

#if KITSUNE_LAMBDA_ENABLED
#include "llvm/Transforms/Tapir/LambdaABI.h"
#endif

#if KITSUNE_OMPTASK_ENABLED
#include "llvm/Transforms/Tapir/OMPTaskABI.h"
#endif

#if KITSUNE_OPENCILK_ENABLED
#include "llvm/Transforms/Tapir/OpenCilkABI.h"
#endif

#if KITSUNE_PTHREADS_ENABLED
#include "kitsune/Targets/PthreadsTT.h"
#endif

#if KITSUNE_QTHREADS_ENABLED
#include "kitsune/Targets/QthreadsTT.h"
#endif

#if KITSNUE_REALM_ENABLED
#include "kitsune/Targets/RealmABI.h"
#endif

#if KITSUNE_SERIAL_ENABLED
#include "kitsune/Targets/SerialTT.h"
#endif

using namespace llvm;

[[noreturn]]
static void fatalTTNotEnabled(TTID tt) {
  emitDiagnostic(DiagID::ErrTTNotEnabled, tt);
  exitOnError();
}

static std::unique_ptr<TapirTarget> makeCudaTT(Module &m,
                                               const TTOptions &tto) {
#if KITSUNE_CUDA_ENABLED
  return std::make_unique<CudaABI>(m, tto);
#else
  fatalTTNotEnabled(TTID::Cuda);
#endif
}

static std::unique_ptr<TapirTarget> makeCustomTT(Module &m,
                                                 const TTOptions &tto) {
#if KITSUNE_CUSTOM_ENABLED
  return std::unique_ptr<TapirTarget>(
      tto.getTTPlugin()->makeTapirTarget(m, tto));
#else
  fatalTTNotEnabled(TTID::Custom);
#endif
}

static std::unique_ptr<TapirTarget> makeHipTT(Module &m, const TTOptions &tto) {
#if KITSUNE_HIP_ENABLED
  return std::make_unique<HipABI>(m, tto);
#else
  fatalTTNotEnabled(TTID::Hip);
#endif
}

static std::unique_ptr<TapirTarget> makeLambdaTT(Module &m,
                                                 const TTOptions &tto) {
#if KITSUNE_LAMBDA_ENABLED
  return std::make_unique<LambdaABI>(m, tto);
#else
  fatalTTNotEnabled(TTID::Lambda);
#endif
}

static std::unique_ptr<TapirTarget> makeOMPTaskTT(Module &m,
                                                  const TTOptions &tto) {
#if KITSUNE_OMPTASK_ENABLED
  return std::make_unique<OMPTask>(m, tto);
#else
  fatalTTNotEnabled(TTID::OMPTask);
#endif
}

static std::unique_ptr<TapirTarget> makeOpenCilkTT(Module &m,
                                                   const TTOptions &tto) {
#if KITSUNE_OPENCILK_ENABLED
  return std::make_unique<OpenCilkABI>(m, tto);
#else
  fatalTTNotEnabled(TTID::OpenCilk);
#endif
}

static std::unique_ptr<TapirTarget> makeOpenMPTT(Module &m,
                                                 const TTOptions &tto) {
  llvm_unreachable("'openmp' tapir target is out of date");
}

static std::unique_ptr<TapirTarget> makePthreadsTT(Module &m,
                                                   const TTOptions &tto) {
#if KITSUNE_PTHREADS_ENABLED
  return std::make_unique<PthreadsTT>(m, tto);
#else
  fatalTTNotEnabled(TTID::Pthreads);
#endif
}

static std::unique_ptr<TapirTarget> makeQthreadsTT(Module &m,
                                                   const TTOptions &tto) {
#if KITSUNE_QTHREADS_ENABLED
  return std::make_unique<QthreadsTT>(m, tto);
#else
  fatalTTNotEnabled(TTID::Qthreads);
#endif
}

static std::unique_ptr<TapirTarget> makeRealmTT(Module &m,
                                                const TTOptions &tto) {
#if KITSUNE_REALM_ENABLED
  return std::make_unique<RealmABI>(m, tto);
#else
  fatalTTNotEnabled(TTID::Realm);
#endif
}

static std::unique_ptr<TapirTarget> makeSerialTT(Module &m,
                                                 const TTOptions &tto) {
#if KITSUNE_SERIAL_ENABLED
  return std::make_unique<SerialTT>(m, tto);
#else
  fatalTTNotEnabled(TTID::Serial);
#endif
}

std::unique_ptr<TapirTarget> llvm::makeTT(TTID tt, Module &m,
                                          const TTOptions &tto) {
  switch (tt) {
  case TTID::Nolo:
    return nullptr;
  case TTID::Cuda:
    return makeCudaTT(m, tto);
  case TTID::Custom:
    return makeCustomTT(m, tto);
  case TTID::Hip:
    return makeHipTT(m, tto);
  case TTID::Lambda:
    return makeLambdaTT(m, tto);
  case TTID::OMPTask:
    return makeOMPTaskTT(m, tto);
  case TTID::OpenCilk:
    return makeOpenCilkTT(m, tto);
  case TTID::OpenMP:
    return makeOpenMPTT(m, tto);
  case TTID::Pthreads:
    return makePthreadsTT(m, tto);
  case TTID::Qthreads:
    return makeQthreadsTT(m, tto);
  case TTID::Realm:
    return makeRealmTT(m, tto);
  case TTID::Serial:
    return makeSerialTT(m, tto);
  }
  llvm_unreachable("makeTT: TTID not handled");
}

bool llvm::isTTEnabled(TTID tt) {
  switch (tt) {
  case TTID::Nolo:
  case TTID::Serial:
    return true;
  case TTID::Cuda:
    return kitCudaEnabled();
  case TTID::Custom:
    return kitCustomEnabled();
  case TTID::Hip:
    return kitHipEnabled();
  case TTID::Lambda:
    return kitLambdaEnabled();
  case TTID::OMPTask:
    return kitOMPTaskEnabled();
  case TTID::OpenCilk:
    return kitOpenCilkEnabled();
  case TTID::OpenMP:
    return kitOpenMPEnabled();
  case TTID::Pthreads:
    return kitPthreadsEnabled();
  case TTID::Qthreads:
    return kitQthreadsEnabled();
  case TTID::Realm:
    return kitRealmEnabled();
  }
  llvm_unreachable("isTTEnabled: TTID not handled");
}
