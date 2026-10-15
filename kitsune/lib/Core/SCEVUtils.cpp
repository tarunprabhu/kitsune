//===- SCEVUtils.cpp - Utilities for SCEV objects -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for SCEV objects.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/SCEVUtils.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

using namespace llvm;

const SCEV *llvm::getNonMatchingOperand(const SCEV *sc, const SCEV *match) {
  assert(sc->operands().size() == 2 && "SCEV must have exactly two operands");

  ArrayRef<const SCEV*> scOps = sc->operands();
  const SCEV *sc0 = scOps[0];
  const SCEV *sc1 = scOps[1];
  if (sc0 == match)
    return sc1;
  else if (sc1 == match)
    return sc0;
  else
    return nullptr;
}
