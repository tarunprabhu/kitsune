//===- PrefetchForDevice.h - Generate dtoh/htod prefetch calls --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generate prefetch calls to initiate movement of data between host and device.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_PREFETCH_FOR_DEVICE_H
#define KITSUNE_TRANSFORMS_PREFETCH_FOR_DEVICE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class TTOptions;

/// \ingroup kitsune
/// This pass generates calls to initiate movement of data between host and
/// device.
class PrefetchForDevicePass : public PassInfoMixin<PrefetchForDevicePass> {
protected:
  const TTOptions &tto;

public:
  PrefetchForDevicePass(const TTOptions &tto) : tto(tto) {}

  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_PREFETCH_FOR_DEVICE_H
