//===- OptznLevel.h - (Yet another) optimization level object --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kitsune-specific optimization level (but not really)
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_OPTZN_LEVEL_H
#define KITSUNE_CORE_OPTZN_LEVEL_H

namespace llvm {

/// The various optimization levels supported by Kitsune.
///
/// NOTE: The levels supported here are no different from the levels supported
/// Kitsune. In principle, one could extend this to have some subtle form of
/// differential optimization levels for host and device code (for the tapir
/// targets that use them - particularly the GPU-centric ones), but that is not
/// really why this class exists.
///
/// The motivation for this is to avoid the inevitable circular dependences that
/// are introduced if we use LLVM's \ref OptimizationLevel object. That is
/// defined in LLVMPasses. That is "reasonable" in the case of LLVM since
/// neither the frontends (clang, flang) nor the middle-ends use it very much
/// outside of constructing the pass pipeline. However, Kitsune records this in
/// the \ref TTOptions object in order to use it in the nested emb-optimize
/// pass. This results in a circular dependence between LLVMPasses and
/// LLVMKitCore - something that we would really like to avoid. Moving the
/// OptimizationLevel class to a different library was attempted - LLVMSupport
/// seemed a natural place for it, but that causes more problems.
///
/// LLVMSupport is built with -Werror=global-constructors where possible. LLVM's
/// OptimizationLevel class is not intended to be instantiated on demand -
/// instead, several static members are provided, each of which represents a
/// single optimization level. Obviously, initializing these requires the use of
/// global constructors. Ono the whole, it does seem like a good idea to avoid
/// the use of global constructors given the well-known static initialization
/// issues in C++. In any case, that makes we would have to put it in a less
/// desirable library, or find some other way of dealing with it.
///
/// We have chosen to just use our own optimization level instance in Kitsune
/// and just convert it to LLVM's OptimizationLevel on demand. It's annoying,
/// but, arguably, less so than having introducing even more cyclic dependencies
/// between LLVM and Kitsune than are already present (see llvm/CMakeLists.txt
/// for a details).
enum class OptznLevel {
  /// No optimizations
  O0,

  /// "Some" optimizations (between -O0 and -O2)
  O1,

  /// "Most" optimizations
  O2,

  /// Like -O2, but includes those that take longer and/or produce larger code,
  /// in an attempt to make the program run faster
  O3,

  /// Like -O2 but with extra optimizations to reduce code size
  Os,

  /// Like -Os, but is more aggessive when it comes to reducing code size
  Oz,
};

} // namespace llvm

#endif // KITSUNE_CORE_OPTZN_LEVEL_H
