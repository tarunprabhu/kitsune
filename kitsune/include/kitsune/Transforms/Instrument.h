//===- Instrument.h - Pass to insert Kitsune's instrumentation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass that inserts Kitsune-specific instrumentation.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_INSTRUMENT_H
#define KITSUNE_TRANSFORMS_INSTRUMENT_H

#include "kitsune/Core/Instrumentation.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Pass that inserts Kitsune-specific instrumentation.
class InstrumentPass : public PassInfoMixin<InstrumentPass> {
private:
  const KitInstrOptions &instrOpts;

public:
  explicit InstrumentPass(const KitInstrOptions &instrOpts)
      : instrOpts(instrOpts) {}

  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_INSTRUMENT_H
