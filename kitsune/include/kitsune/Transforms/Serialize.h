//===- Serialize.h - Serialize certain tapir constructs ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Serialize certain tapir constructs.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_SERIALIZE_H
#define KITSUNE_TRANSFORMS_SERIALIZE_H

#include "kitsune/Core/ModuleAttrs.h"
#include "kitsune/Passes/PassUtils.h"
#include "kitsune/Transforms/PreLowerAnnotate.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Serialize certain tapir constructs.
class SerializePass : public PassInfoMixin<SerializePass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);

  using Requires = std::tuple<PreLowerAnnotatePass>;
  static constexpr auto hasRunAttr = ModuleAttrKind::SerializePass;
};

static_assert(check_pass_dependent<SerializePass>());
static_assert(check_pass_requirable<SerializePass>());

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_SERIALIZE_H
