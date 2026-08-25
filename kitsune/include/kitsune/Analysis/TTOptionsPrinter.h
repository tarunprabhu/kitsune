//==- TTOptionsPrinter.h - Pass that prints the TTOptions object -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass that prints the TTOptions object.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_ANALYSIS_TTOPTIONS_PRINTER_H
#define KITSUNE_ANALYSIS_TTOPTIONS_PRINTER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class TTOptions;

/// \ingroup kitsune
/// Print the TTOptions object to stdout.
class TTOptionsPrinterPass : public PassInfoMixin<TTOptionsPrinterPass> {
private:
  const TTOptions &tto;

public:
  TTOptionsPrinterPass(const TTOptions &tto) : tto(tto) {}

  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);
};

} // namespace llvm

#endif // KITSUNE_ANALYSIS_TTOPTIONS_PRINTER_H
