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
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Support/ErrorHandling.h"
#include "kitsune/Support/TypeTraits.h"
#include "kitsune/Targets/OpenMPTT.h"
#include "kitsune/Targets/PthreadsTT.h"
#include "kitsune/Targets/SerialTT.h"

#if KITSUNE_CUDA_ENABLED
#include "kitsune/Targets/CudaABI.h"
#endif // KITSUNE_CUDA_ENABLED

#if KITSUNE_HIP_ENABLED
#include "kitsune/Targets/HipABI.h"
#endif // KITSUNE_HIP_ENABLED

#if KITSUNE_LAMBDA_ENABLED
#include "llvm/Transforms/Tapir/LambdaABI.h"
#endif // KITSUNE_LAMBDA_ENABLED

#if KITSUNE_OMPTASK_ENABLED
#include "llvm/Transforms/Tapir/OMPTaskABI.h"
#endif // KITSUNE_OMPTASK_ENABLED

#if KITSUNE_OPENCILK_ENABLED
#include "llvm/Transforms/Tapir/OpenCilkABI.h"
#endif // KITSUNE_OPENCILK_ENABLED

#if KITSUNE_QTHREADS_ENABLED
#include "kitsune/Targets/QthreadsTT.h"
#endif // KITSUNE_QTHREADS_ENABLED

#if KITSNUE_REALM_ENABLED
#include "kitsune/Targets/RealmABI.h"
#endif // KITSUNE_REALM_ENABLED

// This is what makes the following `makeTTImpl` function work. Since we only
// include the headers for tapir targets that are enabled we need to forward
// declare the types for the tapir target objects that have not been enabled.
// For the enabled tapir targets, the complete definition will be available in
// the included headers. This incomplete declaration will be superseded by the
// definition. However, for the disabled targets, we will have a valid, but
// incomplete type with the same name as the complete type that would have been
// present if the target were enabled. This ensures that there are no type
// errors in the `if constexpr` statement in `makeTTImpl`. Note that we have
// only included those tapir targets that are not guaranteed to be built.
//
// When adding a new tapir target, a "forward declaration" for the target must
// be added here, if the tapir target object will be created by the `makeTTImpl`
// function, and if it is not guaranteed to be built.
namespace llvm {
class CudaABI;
class HipABI;
class LambdaABI;
class OMPTaskABI;
class OpenCilkABI;
class QthreadsTT;
class RealmABI;
} // namespace llvm

using namespace llvm;

[[noreturn]]
static void fatalTTNotEnabled(TTID tt) {
  emitDiagnostic(DiagID::ErrTTNotEnabled, tt);
  exitOnError();
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

template <typename T>
static std::unique_ptr<TapirTarget> makeTTImpl(TTID tt, Module &m,
                                               const TTOptions &tto) {
  if constexpr (std::is_complete_v<T>)
    return std::make_unique<T>(m, tto);
  else
    fatalTTNotEnabled(tt);
}

std::unique_ptr<TapirTarget> llvm::makeTT(TTID tt, Module &m,
                                          const TTOptions &tto) {
  //
  // ----------- NOTE FOR ANYONE UPDATING THE SWITCH STATEMENT BELOW -----------
  //
  // When adding a case to this switch statement, a forward declaration
  // for the type of the TT object being created must be added to the namespace
  // llvm block above, unless either of the following is true:
  //
  //   - The tapir target is universal, i.e. it is always enabled.
  //
  //   - The tapir target is not created using the `makeTTImpl` helper function.
  //
  // ---------------------------------------------------------------------------

  // clang-format off
  switch (tt) {
  case TTID::Nolo: return nullptr;
  case TTID::Cuda: return makeTTImpl<CudaABI>(tt, m, tto);
  case TTID::Custom: return makeCustomTT(m, tto);
  case TTID::Hip: return makeTTImpl<HipABI>(tt, m, tto);
  case TTID::Lambda: return makeTTImpl<LambdaABI>(tt, m, tto);
  case TTID::OMPTask: return makeTTImpl<OMPTaskABI>(tt, m, tto);
  case TTID::OpenCilk: return makeTTImpl<OpenCilkABI>(tt, m, tto);
  case TTID::OpenMP: return makeTTImpl<OpenMPTT>(tt, m, tto);
  case TTID::Pthreads: return makeTTImpl<PthreadsTT>(tt, m, tto);
  case TTID::Qthreads: return makeTTImpl<QthreadsTT>(tt, m, tto);
  case TTID::Realm: return makeTTImpl<RealmABI>(tt, m, tto);
  case TTID::Serial: return makeTTImpl<SerialTT>(tt, m, tto);
  }
  // clang-format on
  llvm_unreachable("makeTT: TTID not handled");
}
