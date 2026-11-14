//===- EmbLowerIntrinsics.cpp - Lower Kitsune-specific intrinsics ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune-specific intrinsics in an embedded module. Although most
// Kitsune-specific intrinsics are lowered by the backend, lowering of the
// GPU-specific intrinsics is run as part of Kitsune's lowering pipeline in the
// middle-end.
//
// In some cases, the intrinsics lower to LLVM-IR that can be optimized. In such
// cases, we don't strictly need to lower the intrinsics before codegen, but it
// would be sub-optimal to not do so. In other cases, the intrinsics *must* be
// lowered early. For example, when compiling for AMD GPU's, the standard set of
// optimization passes run on the embedded bitcode module includes the
// AMDGPUAttributor pass, which adds target-specific function attributes. This
// includes attributes such as `amdgpu-no-workgroup-id-y` that indicates that
// the function does not query workGroupId.y. There are corresponding attributes
// for `workitem`, `workGroupSize` etc. The attributor pass looks for the
// correct target-specific intrinsic calls to determine whether these parameters
// are queried. If these attributes are added, but the function does query these
// parameters, an invalid value will be returned at runtime.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EmbLowerIntrinsics.h"
#include "LowerIntrinsics.h"

#define DEBUG_TYPE "emb-lower-intrinsics"

using namespace llvm;

bool EmbLowerIntrinsicsPass::run(TTID tt, Module &devM, Module &hostM,
                                 ModuleAnalysisManager &hostAM) {
  detail::IntrinsicKinds kindsToLower = detail::IntrinsicKind::GPUReduce |
                                        detail::IntrinsicKind::GPUIndex |
                                        detail::IntrinsicKind::GPUWarp;
  return detail::LowerIntrinsicsImpl(tto, kindsToLower).runOn(devM);
}
