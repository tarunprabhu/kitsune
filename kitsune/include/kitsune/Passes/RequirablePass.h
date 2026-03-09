//=- RequirablePass.h - Pass that may be required by other passes -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass that may be required by other transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_PASSES_REQUIRABLE_PASS_H
#define KITSUNE_PASSES_REQUIRABLE_PASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Base class for a pass that may be required by a DependentPass. In such
/// cases, this pass should be run before the requiring pass is run. The
/// hasRun() static method from this class must return true if the pass has been
/// run on a module. DependentPass'es that require this pass will use that
/// method to determine if the pass has been run.
///
/// Passes inheriting from this class must provide a definition of this
/// method as shown below:
///
/// \code
///     template<> bool RequirablePass<Derived>::hasRun(const Module &m) {
///         ...
///     }
/// \endcode
///
/// In most cases, requirable passes will write an attribute to a module to
/// indicate that the pass has been run. However, this is not strictly required
/// and passes are free to use any other method to determine if they have been
/// run.
///
/// Passes using the attribute mechanism to indicate that the pass has been run
/// must set the attribute in inheriting pass' run() method. However, this
/// requirement is not enforced. Failing to do so may result in errors being
/// raised by dependent passes when they run.
///
template <typename T> class RequirablePass {
public:
  static bool hasRun(const Module &m);
};

} // namespace llvm

#endif // KITSUNE_PASSES_REQUIRABLE_PASS_H
