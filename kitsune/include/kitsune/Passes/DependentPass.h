//=- DependentPass.h - Pass that requires other pass to have run --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass that requires other passes to have run.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_PASSES_DEPENDENT_PASS_H
#define KITSUNE_PASSES_DEPENDENT_PASS_H

#include "kitsune/Frontend/Diagnostics.h"
#include "kitsune/Passes/RequirablePass.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Base class for a pass that depends on other passes. This is very different
/// from a pass that requires analyses. This is to be used to declare
/// dependencies between transformation passes.
///
/// An example of a dependent pass is the kit-serialize pass. This examines
/// annotations on tapir loops, and therefore, requires the
/// kit-annotate-tapir-loops pass to have been run first. The serialize pass
/// will not necessarily fail if the annotator pass has not been run. But it
/// may have no effect. In principle, this may cause later passes to fail.
///
/// Consider the concrete example of compiling a tapir loop nest for a
/// GPU-centric tapir target, the maximum depth of parallel loops can be at
/// most 3. The serialize pass will serialize any parallel loops at depths
/// greater than 3. If this is not done prior before the loop spawning pass is
/// run, the latter will fail.
///
/// The example below shows how the serialize pass may be declared in this
/// case.
///
/// \code{.cpp}
///     class SerializePass
///         : public PassInfoMixin<SerializePass>,
///           public DependentPass<SerializePass, AnnotateTapirLoopsPass> {
///       ...
///     };
/// \endcode
///
/// In this case, `SerializePass` is a "dependent" pass while
/// `AnnotateTapirLoopsPass` is a "requirable" pass. Requirable passes must
/// inherit from the RequirablePass base class.
///
/// Dependent passes must call the checkReqdPassesHaveRun() method early in the
/// the run() methods to ensure that the required passes have run. Currently,
/// there is no way to ensure that this happens automatically, nor is there
/// any way to check that developers do so themselves.
///
template <typename Pass, typename... Requires> class DependentPass {
private:
  // Compile-time function that checks that a pass is requirable.
  template <typename T> static constexpr bool isRequirable() {
    constexpr bool requirable = std::is_base_of_v<RequirablePass<T>, T>;
    static_assert(requirable, "Required pass must define a hasRun method");
    return requirable;
  }

  // Compile-time function that checks that all required passes are
  // requirable.
  template <typename T, typename... Ts> static constexpr bool allRequirable() {
    bool requirable = isRequirable<T>();
    if constexpr (sizeof...(Ts))
      return requirable && allRequirable<Ts...>();
    return requirable;
  }

  // Ensure that all required passes are requirable.
  static_assert(allRequirable<Requires...>(),
                "All passes required by a dependent pass must be requirable");

  // Check that a required pass has run. If it has not, a diagnostic will be
  // emitted to stderr.
  template <typename P, typename T>
  static bool reqdPassHasRun(const Module &m) {
    bool hasRun = T::hasRun(m);
    if (!hasRun)
      emitDiagnostic(DiagID::ErrRequiredPassNotRun, P::name(), T::name());
    return hasRun;
  }

  // Returns a count of the number of required passes that have *not* been run.
  template <typename P, typename T, typename... Ts>
  static unsigned reqdPassesNotRun(const Module &m) {
    unsigned notRun = reqdPassHasRun<P, T>(m) ? 0 : 1;
    if constexpr (sizeof...(Ts))
      return notRun + reqdPassesNotRun<P, Ts...>(m);
    return notRun;
  }

protected:
  /// Check that all required passes have been run. A diagnostic message will
  /// be emitted on stderr for every required pass that has not been run. If at
  /// least one pass has been run, this will cause the program to exit
  /// gracefully with a system-dependent error code.
  static void checkReqdPassesHaveRun(const Module &m) {
    if (reqdPassesNotRun<Pass, Requires...>(m))
      exitOnError();
  }
};

} // namespace llvm

#endif // KITSUNE_PASSES_DEPENDENT_PASS_H
