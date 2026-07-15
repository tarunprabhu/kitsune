//===- GenerateCtorsCommon.h - Common code for ctor generation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Code shared by the ctor generators (for CPU-centric and GPU-centric tapir
// targets).
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_TRANSFORMS_GENERATE_CTORS_COMMON_H
#define KITSUNE_LIB_TRANSFORMS_GENERATE_CTORS_COMMON_H

#include "kitsune/Core/Tapir.h"
#include "llvm/IR/IRBuilder.h"

#define DEBUG_TYPE "kit-ctors"

namespace llvm {

class Function;
class Module;
class TTOptions;

namespace detail {

// The priority must be in the range [101,65535] with larger values having
// lower priority relative to other global constructors in @llvm.global_ctors
// (respectively destructors in @llvm.global_dtors).
static constexpr unsigned kitCtorPriority = 65535;
static constexpr unsigned kitDtorPriority = 65535;

/// Options to generate global ctors for kitsune's runtime.
struct GenerateCtorOptions {
public:
  /// Launch kernel using Y-axis threading.
  unsigned useYLaunch : 1;
};

/// Abstract base class for all ctor/dtor generators.
class GenerateCtorBase {
protected:
  TTID tt;
  const TTOptions &tto;

protected:
  GenerateCtorBase(TTID tt, const TTOptions &tto);

  /// Get an IRBuilder<> to use with a functional skeleton. The skeleton is
  /// expected to have a single entry and exit block with an unconditional
  /// branch between the entry and exit blocks and a single return instruction
  /// in the exit block. The insertion point of the builder will be the start
  /// of the entry block.
  IRBuilder<> getBuilderForSkeleton(Function *f);

  /// Generates a skeleton of the ctor. This will consist of an entry and exit
  /// block. The entry block will only contain a single unconditional branch to
  /// the exit block. The exit block will contain a single return instruction.
  Function *genCtorSkeleton(Module &m);

  /// Generates a skeleton of the dtor. This will consist of an entry and exit
  /// block. The entry block will only contain a single unconditional branch to
  /// the exit block. The exit block will contain a single return instruction.
  Function *genDtorSkeleton(Module &m);

public:
  virtual ~GenerateCtorBase() = default;
  virtual void run(Module &m) = 0;
};

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_GENERATE CTORS_COMMON_H
