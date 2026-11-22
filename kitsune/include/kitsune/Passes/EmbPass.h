//===- EmbPass.h - Base classes for embedded module passes ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for passes that operate on embedded modules. These typically
// perform transformations on the embedded modules and update the global
// variables in the parent module that contain them.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_PASSES_EMB_PASS_H
#define KITSUNE_PASSES_EMB_PASS_H

#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/GVAttrs.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TTID.h"
#include "kitsune/Passes/NestedPipelineContext.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

namespace detail {

// Signature of the run function in an embedded bitcode pass that requires an
// analysis manager for the embedded module (the host module always has a
// corresponding analysis manager).
template <typename EmbPass>
using run_with_manager_t =
    bool (EmbPass::Pass::*)(TTID, typename EmbPass::Entity &devE,
                            AnalysisManager<typename EmbPass::Entity> &devAM,
                            Module &hostM, AnalysisManager<Module> &hostAM);

// SFINAE helper class to determine if an embedded bitcode pass has a run method
// that requires an analysis manager for the embedded module.
template <typename EmbPass, typename = void>
struct has_run_with_manager_t : std::false_type {};

// SFINAE helper class to determine if an embedded bitcode pass has a run method
// that requires an analysis manager for the embedded module.
template <typename EmbPass>
struct has_run_with_manager_t<
    EmbPass, std::void_t<decltype(static_cast<run_with_manager_t<EmbPass>>(
                 &EmbPass::Pass::run))>> : std::true_type {};

// Trait that checks if an embedded bitcode pass requires an analysis manager
// for the embedded module.
template <typename EmbPass>
static constexpr bool has_run_with_manager_v =
    has_run_with_manager_t<EmbPass>::value;

//==------------------------------------------------------------------------==//

// Signature of the run function in an embedded bitcode pass that does not
// require an analysis manager for the embedded module (the host module always
// has a corresponding analysis manager).
template <typename EmbPass>
using run_without_manager_t =
    bool (EmbPass::Pass::*)(TTID, typename EmbPass::Entity &devE, Module &hostM,
                            AnalysisManager<Module> &hostAM);

// SFINAE helper class to determine if an embedded bitcode pass has a run method
// that does not require an analysis manager for the embedded module.
template <typename EmbPass, typename = void>
struct has_run_without_manager_t : std::false_type {};

// SFINAE helper class to determine if an embedded bitcode pass has a run method
// that does not require an analysis manager for the embedded module.
template <typename EmbPass>
struct has_run_without_manager_t<
    EmbPass, std::void_t<decltype(static_cast<run_without_manager_t<EmbPass>>(
                 &EmbPass::Pass::run))>> : std::true_type {};

// Trait that checks if an embedded bitcode pass does not require an analysis
// manager for the embedded module.
template <typename EmbPass>
static constexpr bool has_run_without_manager_v =
    has_run_without_manager_t<EmbPass>::value;

//==------------------------------------------------------------------------==//

// CRTP Base class for a pass that operates on an entity in an embedded bitcode
// module. This is inherited by classes that operate on specific units in the
// embedded module. See, for instance \ref EmbModulePass, and
// \ref EmbFunctionPass.
//
// \tparam EntityPassT is the type of the class that inherits this
// class. \tparam EntityPassT must provide two runImpl methods whose signatures
// are as follows:
//
//     bool runImpl(TTID,
//                  Module& devM,
//                  Module& hostM,
//                  ModuleAnalysisManager& hostMAM);
//
//     bool runImpl(TTID,
//                  Module& devM,
//                  ModuleAnalysisManager& devMAM,
//                  Module& hostM,
//                  ModuleAnalysisManager& hostMAM);
//
// The run method provided by this class will be called by LLVM's standard pass
// manager. This in turn will call the runImpl methods of the derived class.
// The derived classes are expected to operate on specific entities such as
// functions, modules, or loops.
//
// \tparam EntityT is the unit on which the embedded pass operates. For
// instance, if \tparam EntityT is \ref Module, the embedded bitcode pass,
// \tparam PassT will act like a standard module pass, but on the embedded
// module. Similarly, if \tparam is Function, the embedded bitcode pass
// \tparam PassT will operate on functions in the embedded bitcode module.
//
// \tparam PassT is the actual embedded bitcode pass. That pass is expected to
// provided *exactly* one method named run() method with one of the following
// signatures:
//
//     bool run(TTID,
//              EntityT& devE,
//              Module& hostM,
//              ModuleAnalysisManager& hostMAM);
//
//   OR
//
//     bool run(TTID,
//              EntityT& devE,
//              AnalysisManager<Entity>& devAM,
//              Module& hostM,
//              ModuleAnalysisManager& hostMAM);
//
// The former is used if the pass does not require any analyses on the embedded
// module, while the latter is used if the pass may need analyses. While it is
// safe to always use the latter, it may be inefficient since it will involve
// the creation and initialization of several analysis managers unnecessarily.
//
// NOTE: Embedded bitcode passes *MUST NOT* modify the host module. They may
// examine the host module and request analyses from it. The only modification
// these passes may make to the host is to update the initializer of the global
// variable containing the bitcode that was modified.
//
template <template <typename> class EntityPassT, typename PassT,
          typename EntityT>
class EmbPass : public PassInfoMixin<PassT> {
public:
  using Pass = PassT;
  using Entity = EntityT;

public:
  // Since this inherits from PassInfoMixin, this is the function that is
  // invoked by LLVM's standard pass manager.
  PreservedAnalyses run(Module &hostM, ModuleAnalysisManager &hostMAM) {
    using This = EmbPass<EntityPassT, PassT, EntityT>;
    using EntityPass = EntityPassT<PassT>; // This is the derived class

    // We have to add the static assertion here because this is the first time
    // that PassT is complete. If we add this assertion into the body of the
    // class (which is the more natural place for it), PassT will be incomplete
    // and the assertion will always fail.
    constexpr bool runWithManager = detail::has_run_with_manager_v<This>;
    constexpr bool runWithoutManager = detail::has_run_without_manager_v<This>;
    static_assert(runWithManager != runWithoutManager,
                  "Embedded pass must have exactly one run method");

    for (GlobalVariable *g : getGlobals(hostM, GVAttrKind::BitCode)) {
      Expected<std::unique_ptr<Module>> devMOrErr = parseEmbBCGlobal(*g);
      if (not devMOrErr)
        exitOnError(devMOrErr.takeError());

      bool changed = false;
      std::unique_ptr<Module> devM = std::move(devMOrErr.get());
      TTID tt = *getBitCodeAttr(*g);
      EntityPass *entityPass = static_cast<EntityPass *>(this);
      if constexpr (runWithManager) {
        NestedPipelineContext pipelineCtx;
        ModuleAnalysisManager &devMAM = pipelineCtx.getModuleAnalysisManager();
        changed |= entityPass->runImpl(tt, *devM, devMAM, hostM, hostMAM);
      } else {
        changed |= entityPass->runImpl(tt, *devM, hostM, hostMAM);
      }

      if (changed)
        resetEmbBCGlobal(*devM, *g);
    }

    // This will preserve all analyses because the only thing that may have
    // changed are the types and initializers of one or more global variables.
    return PreservedAnalyses::all();
  }
};

} // namespace detail

/// \addtogroup kitsune
/// @{

/// CRTP base class for embedded bitcode passes that operate on embedded
/// modules.
///
/// The classes that inherit from this base class are the actual passes that
/// will operate on the embedded bitcode module. These must provide *exactly*
/// one function named run() with either of the following signatures:
///
///     bool run(TTID,
///              Module& devM,
///              Module& hostM,
///              ModuleAnalysisManager& hostMAM);
///
///   OR
///
///     bool run(TTID,
///              Module& devM,
///              ModuleAnalysisManager& devMAM,
///              Module& hostM,
///              ModuleAnalysisManager& hostMAM);
///
/// The former is used if the pass does not require any analyses on the
/// embedded bitcode module, while the latter is used if the pass may need
/// analyses. While it is safe to always use the latter, it may be inefficient
/// since it will involve the creation and initialization of several analysis
/// managers unnecessarily.
///
/// NOTE: Embedded bitcode passes *MUST NOT* modify the host module. They
/// may examine the host module and request analyses from it. The only
/// modification these passes may make to the host is to update the initializer
/// of the global variable containing the bitcode that was modified.
///
template <typename PassT>
class EmbModulePass : public detail::EmbPass<EmbModulePass, PassT, Module> {
  friend class detail::EmbPass<EmbModulePass, PassT, Module>;

protected:
  bool runImpl(TTID tt, Module &devM, Module &hostM,
               ModuleAnalysisManager &hostMAM) {
    return static_cast<PassT *>(this)->run(tt, devM, hostM, hostMAM);
  }

  bool runImpl(TTID tt, Module &devM, ModuleAnalysisManager &devMAM,
               Module &hostM, ModuleAnalysisManager &hostMAM) {
    return static_cast<PassT *>(this)->run(tt, devM, devMAM, hostM, hostMAM);
  }
};

/// CRTP base class for embedded bitcode passes that operate on defined
/// functions in embedded modules.
///
/// The classes that inherit from this base class are the actual passes that
/// will operate on the functions in the embedded module. These must provide
/// *exactly* one function named run() with either of the following signatures:
///
///     bool run(TTID,
///              Function& devF,
///              Module& hostM,
///              ModuleAnalysisManager& hostMAM);
///
///   OR
///
///     bool run(TTID,
///              Function& devF,
///              FunctionAnalysisManager& devFAM,
///              Module& hostM,
///              ModuleAnalysisManager& hostMAM);
///
/// The former is used if the pass does not require any analyses on the
/// functions in the embedded module, while the latter is used if the pass may
/// need analyses. While it is safe to always use the latter, it may be
/// inefficient since it will involve the creation and initialization of several
/// analysis managers unnecessarily.
///
/// NOTE: Embedded bitcode passes *MUST NOT* modify the host module. They
/// may examine the host module and request analyses from it. The only
/// modification these passes may make to the host is to update the initializer
/// of the global variable containing the bitcode that was modified.
///
template <typename PassT>
class EmbFunctionPass
    : public detail::EmbPass<EmbFunctionPass, PassT, Function> {
  friend class detail::EmbPass<EmbFunctionPass, PassT, Function>;

protected:
  bool runImpl(TTID tt, Module &devM, Module &hostM,
               ModuleAnalysisManager &hostMAM) {
    PassT *pass = static_cast<PassT *>(this);
    bool changed = false;
    for (Function &devF : devM)
      if (devF.size())
        changed |= pass->run(tt, devF, hostM, hostMAM);
    return changed;
  }

  bool runImpl(TTID tt, Module &devM, ModuleAnalysisManager &devMAM,
               Module &hostM, ModuleAnalysisManager &hostMAM) {
    PassT *pass = static_cast<PassT *>(this);
    auto &proxy = devMAM.getResult<FunctionAnalysisManagerModuleProxy>(devM);
    FunctionAnalysisManager &devFAM = proxy.getManager();
    bool changed = false;
    for (Function &devF : devM)
      if (devF.size())
        changed |= pass->run(tt, devF, devFAM, hostM, hostMAM);
    return changed;
  }
};

/// @}

} // namespace llvm

#endif // KITSUNE_PASSES_EMB_PASS_H
