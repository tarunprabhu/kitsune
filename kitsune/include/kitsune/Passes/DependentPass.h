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
///
///     class SerializePass
///         : public DependentPass<SerializePass, AnnotateTapirLoopsPass> {
///        ...
///     };
///
/// \endcode
///
/// In this case, `SerializePass` is a "dependent" pass while
/// `AnnotateTapirLoopsPass` is a "requireable" pass.
///
/// Requireable passes must provide some way to check if they have been run.
/// In most cases, passes that may be required by others will add an attribute
/// to the module being compiled. The presence of this attribute will indicate
/// that the pass has been run. However, passes may use some other mechanism.
///
/// Requireable passes must also provide a static hasRun method with the
/// following signature.
///
/// \code{.cpp}
///
///     static bool hasRun(const llvm::Module &m);
///
/// \endcode
///
/// This method is called to determine if all required passes have been run.
///
/// Dependent passes must call the checkReqdPassesHaveRun() method early in the
/// the run() methods to ensure that the required passes have run. Currently,
/// there is no way to ensure that this happens automatically, nor is there
/// any way to check that developers do so themselves.
///
template <typename Pass, typename... Requires>
class DependentPass : public PassInfoMixin<Pass> {
private:
  // SFINAE helper classes to check if a pass is requireable. A pass is
  // requireable if it defines a static hasRun method.
  //
  // TODO: These checks should be made stronger. We probably also want to check
  // the following:
  //
  //   - The signature of the hasRun method - not just its presence.
  //   - The class should inherit from PassInfoMixin.
  //
  template <typename T, typename = void>
  struct is_requireable_pass : std::false_type {};

  template <typename T>
  struct is_requireable_pass<T, std::void_t<decltype(T::hasRun)>>
      : std::true_type {};

  // Compile-time function that checks that a pass is requireable.
  template <typename T> static constexpr bool isRequireable() {
    constexpr bool requireable = is_requireable_pass<T>::value;
    static_assert(requireable, "Required pass must define a hasRun method");
    return requireable;
  }

  // Compile-time function that checks that all required passes are
  // requireable.
  template <typename T, typename... Ts> static constexpr bool allRequireable() {
    bool requireable = isRequireable<T>();
    if constexpr (sizeof...(Ts))
      return requireable && allRequireable<Ts...>();
    return requireable;
  }

  // Ensure that all required passes are requireable.
  static_assert(allRequireable<Requires...>(),
                "All passes required by a dependent pass must be requireable");

  // Check that a required pass has run. If it has not, a diagnostic will be
  // emitted to stderr.
  template <typename P, typename T>
  static bool reqdPassHasRun(const Module &m) {
    bool hasRun = T::hasRun(m);
    if (!hasRun)
      emitDiagnostic(DiagID::ErrRequiredPassNotRun, P::name(), T::name());
    return hasRun;
  }

  // Implementation function that checks that all required passes have been
  // run.
  template <typename P, typename T, typename... Ts>
  static bool allReqdPassesHaveRun(const Module &m) {
    bool hasRun = reqdPassHasRun<P, T>(m);
    if constexpr (sizeof...(Ts))
      return hasRun && allReqdPassesHaveRun<P, Ts...>(m);
    return hasRun;
  }

protected:
  /// Check that all required passes have been run. A diagnostic message will
  /// be emitted on stderr for every required pass that has not been run. If at
  /// least one pass has been run, this will cause the program to exit
  /// gracefully with a system-dependent error code.
  static void checkReqdPassesHaveRun(const Module &m) {
    if (!allReqdPassesHaveRun<Pass, Requires...>(m))
      exitOnError();
  }
};

} // namespace llvm

#endif // KITSUNE_PASSES_DEPENDENT_PASS_H
