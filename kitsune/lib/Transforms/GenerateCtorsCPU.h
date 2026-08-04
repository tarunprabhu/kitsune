//===- GenerateCtorsCPU.h - Generate ctor for CPU tapir targets -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic implementation of ctor/dtor generation for CPU-centric tapir targets.
// This is usually sufficient for most of these tapir targets.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_TRANSFORMS_GENERATE_CTORS_CPU_H
#define KITSUNE_LIB_TRANSFORMS_GENERATE_CTORS_CPU_H

#include "GenerateCtorsCommon.h"

namespace llvm {

namespace detail {

/// Base class to generate ctors/dtors for CPU-centric tapir targets.
class GenerateCtorCPU : public GenerateCtorBase {
protected:
  /// Generate the ctor. It is the caller's responsibility to append the
  /// returned function to \@llvm.global_ctors.
  virtual Function *genCtor(Module &m);

  /// Generate the dtor. It is the caller's responsibility to append the
  /// returned function to \@llvm.global_dtors.
  virtual Function *genDtor(Module &m);

public:
  GenerateCtorCPU(TTID tt, const TTOptions &tto);
  virtual ~GenerateCtorCPU() = default;

  virtual void run(Module &m) override;
};

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_GENERATE_CTORS_CPU_H
