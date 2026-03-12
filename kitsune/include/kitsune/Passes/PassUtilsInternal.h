//===- PassUtilsInternal.h - Utilities for pass dependence ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to support dependent (and requirable) passes.
//
//   - This *SHOULD NOT* be merged into PassUtils.h.
//
//   - This *SHOULD NOT* be used directly anywhere except in
//     llvm/IR/PassManagerInternal.h.
//
// Changes to this will trigger a pretty large rebuild. It is unlikely that this
// file will change much, so it's probably best to keep it separate from
// PassUtils.h which is likely to change more frequently.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_PASSES_PASS_UTILS_INTERNAL_H
#define KITSUNE_PASSES_PASS_UTILS_INTERNAL_H

#include "kitsune/Frontend/Diagnostics.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/STLExtras.h"

namespace llvm {

namespace detail {

// Trait to get the IRUnit type of the run method of the pass.
template <typename T> struct pass_run_traits;
template <typename ReturnT, typename ClassT, typename IRUnitT, typename... Args>
struct pass_run_traits<ReturnT (ClassT::*)(IRUnitT, Args...)> {
  using irunit_t = IRUnitT;
};

// Trait to get the IRUnit of a pass.
template <typename PassT>
using pass_irunit_t = typename pass_run_traits<decltype(&PassT::run)>::irunit_t;

// Helper to check if a pass has a member with the following signature:
//
//     void setHasRun(IRUnitT&);
//
// Here, IRUnitT is the IR unit on which the pass operates.
template <typename T>
using has_method_record_t =
    decltype(declval<T &>().setHasRun(declval<pass_irunit_t<T>>()));
template <typename T>
constexpr bool has_method_record_v = is_detected<has_method_record_t, T>::value;

// Helper to check if a pass has a member with the following signature:
//
//     static bool setHasRun(const IRUnitT&);
//
// Here, IRUnitT is the IR unit on which the pass operates.
template <typename T>
using has_method_check_t =
    decltype(T::hasRun(declval<const pass_irunit_t<T>>()));
template <typename T>
constexpr bool has_method_check_v = is_detected<has_method_check_t, T>::value;

// Convience helper to check if a pass is requirable.
template <typename T>
constexpr bool pass_requirable_v =
    has_method_record_v<T> && has_method_check_v<T>;

// Helper that checks that all types in a tuple are requirable passes.
template <typename... Ts> struct all_requirable_t : std::false_type {};
template <typename... Ts>
struct all_requirable_t<std::tuple<Ts...>>
    : std::bool_constant<(pass_requirable_v<Ts> && ...)> {};
template <typename Requires>
constexpr bool all_requirable_v = all_requirable_t<Requires>::value;

// Helper that checks if a type is a std::tuple.
template <typename... Ts> struct is_tuple_t : std::false_type {};
template <typename... Ts>
struct is_tuple_t<std::tuple<Ts...>> : std::true_type {};
template <typename T> constexpr bool is_tuple_v = is_tuple_t<T>::value;

// Helper that checks if a type has a member type named Requires.
template <typename T, typename = void>
struct has_member_requires_t : std::false_type {};
template <typename T>
struct has_member_requires_t<T, std::void_t<typename T::Requires>>
    : std::true_type {};
template <typename T>
constexpr bool has_member_requires_v = has_member_requires_t<T>::value;

// Helper function to check if a pass is dependent.
template <typename PassT> static constexpr bool pass_dependent_impl() {
  if constexpr (detail::has_member_requires_v<PassT>)
    return detail::is_tuple_v<typename PassT::Requires> &&
           std::tuple_size_v<typename PassT::Requires> &&
           detail::all_requirable_v<typename PassT::Requires>;
  return false;
}

// Helper to check that all elements of a tuple are requirable passes.
template <typename... Ts> struct check_all_requirable;
template <typename... Ts> struct check_all_requirable<std::tuple<Ts...>> {
private:
  template <typename P, typename... Ps> static constexpr bool check() {
    constexpr bool ok = pass_requirable_v<P>;
    static_assert(ok, "Dependency of pass must be requirable");
    if constexpr (sizeof...(Ps))
      return ok && check<Ps...>();
    return ok;
  }

public:
  static constexpr bool run() { return check<Ts...>(); }
};

// Helper class that unpacks the tuple of required passes. The check method can
// be used to check that each pass has run. It will return 0 if all passes have
// run, non-zero otherwise.
template <typename PassT, typename IRUnitT, typename... Ts> struct passes_run;
template <typename PassT, typename IRUnitT, typename... Ts>
struct passes_run<PassT, IRUnitT, std::tuple<Ts...>> {
private:
  // Check if the pass has been run. Return 1 if it has. Otherwise, emit a
  // diagnostic and return 1.
  template <typename T> static unsigned countPassIfRun(const IRUnitT &ir) {
    if (T::hasRun(ir))
      return 1;
    emitDiagnostic(DiagID::ErrRequiredPassNotRun, PassT::name(), T::name());
    return 0;
  }

  template <typename P, typename... Ps>
  static unsigned countPassesRun(const IRUnitT &ir) {
    unsigned run = countPassIfRun<P>(ir);
    if constexpr (sizeof...(Ps))
      return run + countPassesRun<Ps...>(ir);
    return run;
  }

public:
  static unsigned countRun(const IRUnitT &ir) {
    return countPassesRun<Ts...>(ir);
  }
};

} // namespace detail

/// Check that a pass is "requirable". This is true only if the pass defines a
/// member function with the following members:
///
/// \code{.cpp}
///
///     class Pass : public PassInfoMixin<Pass> {
///         ...
///         void setHasRun(IRUnit& ir);
///         static bool hasRun(const IRUnit& ir);
///     };
///
/// \endcode
///
/// Here, IRUnitT& is the unit of IR on which the pass operates. In practice,
/// in Kitsune, this is most likely to be an llvm::Module.
///
template <typename PassT>
constexpr bool pass_requirable_v =
    detail::has_method_check_v<PassT> && detail::has_method_record_v<PassT>;

/// Run checks for a pass that is a requirable pass. When defining a requirable
/// pass, this should be used as follows:
///
/// \code{.cpp}
///
///     class Pass : public PassInfoMixin<Pass> {
///         ...
///         void setHasRun(IRUnit& ir);
///         static bool hasRun(const IRUnit& ir);
///     };
///
///     static_assert(check_pass_requirable<Pass>());
///
/// \endcode
///
/// The assertion should be added after the pass. If the assertion is added in
/// the definition, some members may not be detected and spurious errors may be
/// raised.
///
/// The difference between this is the and the \ref pass_requirable_v trait is
/// that this will emit error messages providing reasons why the pass is not
/// requirable. Generally, this should only be used where the pass is defined.
///
template <typename PassT> static constexpr bool check_pass_requirable() {
  static_assert(detail::has_method_check_v<PassT>,
                "Requirable pass requires static member "
                "`bool hasRun(const llvm::Module&)`");
  static_assert(detail::has_method_record_v<PassT>,
                "Requirable pass requires non-static member "
                "`void setHasRun(llvm::Module&)`");
  return true;
}

/// SFINAE helper that checks if a pass is a "dependent" pass. This is true only
/// if all of the following conditions are satisfied.
///
///   - The pass contains a type member named Requires.
///   - Requires is of type std::tuple.
///   - The tuple contains at least one member.
///   - Every member of the tuple Requires is a requirable pass.
///   - The required passes must operate on the same IR unit as the requiring
///     passes.
///
/// \code{.cpp}
///
///     class Pass : public PassInfoMixin<Pass> {
///         ...
///         using Module = std::tuple<Req1, Req2, ...>;
///     };
///
/// \endcode
template <typename PassT>
constexpr bool pass_dependent_v = detail::pass_dependent_impl<PassT>();

/// Run checks for a pass that is a dependent pass. When defining a dependent
/// pass, this should be used as follows:
///
/// \code{.cpp}
///
///      class Pass : public PassInfoMixin<Pass> {
///          ...
///          using Requires = std::tuple<Req1, Req2, ...>;
///      };
///      static_assert(check_pass_dependent<Pass>());
///
/// \endcode
///
/// The assertion should be added after the pass. If the assertion is added in
/// the definition, some members may not be detected and spurious errors may be
/// raised.
///
/// The difference between this is the and the \ref pass_dependent_v trait is
/// that this will emit error messages providing reasons why the pass is not
/// dependent. Generally, this should only be used where the pass is defined.
///
template <typename PassT> static constexpr bool check_pass_dependent() {
  static_assert(detail::has_member_requires_v<PassT>,
                "Dependent pass requires type member named Requires");
  static_assert(detail::is_tuple_v<typename PassT::Requires>,
                "Type member named Requires in dependent pass must be of type "
                "std::tuple");
  static_assert(std::tuple_size_v<typename PassT::Requires> > 0,
                "Type member named Requires must specify at least one pass");
  static_assert(detail::check_all_requirable<typename PassT::Requires>::run(),
                "All dependencies of pass must be requirable");
  return true;
}

/// For a given dependent pass, check if all required passes have been run. If
/// at least one pass has not been run, a diagnostic will be written to stderr,
/// and execution will be terminated with a system-dependent error code. If the
/// pass is not dependent, this has no effect. \p ir is the IR unit on which
/// the pass operates.
template <typename PassT, typename IRUnitT>
void checkRequiredPassesHaveRun(const IRUnitT &ir) {
  if constexpr (pass_dependent_v<PassT>) {
    static_assert(std::is_same_v<IRUnitT &, detail::pass_irunit_t<PassT>>,
                  "IR type mismatch");
    using Requires = typename PassT::Requires;
    unsigned run = detail::passes_run<PassT, IRUnitT, Requires>::countRun(ir);
    if (run != std::tuple_size_v<Requires>)
      exitOnError();
  }
}

/// If the given pass, \p pass is a requirable pass, record the fact that the
/// pass has been run. This will call the setHasRun method on the pass. This
/// method may modify the give IR unit, \p ir, but that is not strictly
/// required. The pass may use any other method to keep track of whether or not
/// it has run. Regardless of the method used, subsequent calls to the
/// hasRun() method of the pass should return `true`. If the pass is not
/// requirable, this has no effect.
template <typename PassT, typename IRUnitT>
void setPassHasRun(PassT &pass, IRUnitT &ir) {
  if constexpr (pass_requirable_v<PassT>) {
    static_assert(std::is_same_v<IRUnitT &, detail::pass_irunit_t<PassT>>,
                  "IR type mismatch");
    pass.setHasRun(ir);
  }
}

} // namespace llvm

#endif // KITSUNE_PASSES_PASS_UTILS_INTERNAL_H
