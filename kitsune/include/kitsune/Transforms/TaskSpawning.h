//===- TaskSpawning.h - Spawn tasks ----------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Spawn tasks. This is the equivalent of the loop-spawning pass, but for
// "tasks", i.e. Single Entry Single Exit regions demarcated by detach and
// reattach instructions that are not part of a tapir loop.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_TASK_SPAWNING_H
#define KITSUNE_TRANSFORMS_TASK_SPAWNING_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Spawn tasks.
class TaskSpawningPass : public PassInfoMixin<TaskSpawningPass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_TASK_SPAWNING_H
