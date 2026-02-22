//===- TapirTargets.h - Utilities to create tapir targets ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to create tapir target objects.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_TAPIR_TARGETS_H
#define KITSUNE_CORE_TAPIR_TARGETS_H

#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {

/// Create a tapir tapir target object.
///
/// \param tt   The tapir target ID
/// \param m    The module. In the case of tapir targets that compile for
///             multiple devices, such as the GPU-centric tapir targets, this is
///             the host module.
/// \param tto  The tapir target options object
std::unique_ptr<TapirTarget> makeTT(TTID tt, Module &m, const TTOptions &tto);

} // namespace llvm

#endif // KITSUNE_CORE_TAPIR_TARGETS_H
