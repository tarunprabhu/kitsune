//===- LocalizeGlobals.h - Passes globals explicitly to users  -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that groups all the global variables used by
// each function in the module in to a single struct and passes that struct
// explicitly as an additional argument to the function.
//
//===----------------------------------------------------------------------===//

#ifndef TapirLOCALIZE_GLOBALS_H
#define TapirLOCALIZE_GLOBALS_H

#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Type.h"

#include <map>
#include <vector>

namespace llvm {

// The actual implementation class for each function.
class LocalizeGlobalsImpl;

// Maintains the state needed to carry out the localization of globals. This was
// originally written for the Cuda backend, but is general enough to be of use
// by other backends. The terminology used in this class methods reflect this.
//
//     device function  The function into which the globals are localized.
//
//     device global    The global variable used by a device function. This is
//                      a reference to the host global that is declared in
//                      the device module and defined in the host module.
//
//     device module    The module containing the device function.
//
//     host function    The function that calls the device function (there may
//                      be more than one host function that calls a given
//                      device function).
//
//     host global      The GlobalVariable in the host that is used by a
//                      device function. This is defined in the host module.
//
//     host module      The module containing the host function.
//
//     global closure   The structure containing the values of all the global
//                      variables that are used by the device.
//
//     closure type     The llvm::StructType of the global closure object.
//
// The global variables used in the device function are assumed to have been
// defined in the host module (one can assume that the top-level entities i.e.
// function and globals in the device module were cloned from the host).
// This class assumes that the host and device functions are in separate
// modules but does not require it and will work even if the host and device
// modules are one and the same.
//
// ASSUMPTIONS
//
//     - Even if the host and device modules are different, they share the
//       same LLVMContext object. This means that types are shared between
//       the two modules. Therefore, only one closure type needs to be created,
//       separate, but structurally identical types for the host and device
//       modules are not needed.
//
// BACKGROUND: This was originally developed to allow programs with an
// explicit parallel for loop (it doesn't matter how they were automatically
// determined to be parallel for's or were explicitly declared as such) to be
// compiled for an NVIDIA GPU even if they were originally written as CPU code.
// This was done by compiling these loops to Cuda kernels.
// Since the original code was written for the CPU, some used global variables
// that caused problems since GPU's cannot automatically access CPU memory. The
// alternatives were to use the CUDA driver API to register global variables
// with the GPU, use unified memory to allow the globals to be shared, or to
// explicitly pass the global variables to the GPU. This class enables the
// last of those approaches.
//
// FIXME: One major limitation of the current implementation is that it does
// not support device functions calling other device functions. In order to
// enable this, any device function must be passed the transitive closure of all
// global variables used transitively in the function i.e. by its callees.
//
class LocalizeGlobals {
public:
  enum Mode {
    // The global closure is passed by value for locally-const globals and by
    // reference otherwise.
    ValueStruct,

    // The global closure is always passed by reference.
    RefStruct,

    // The globals are not combined into a single closure. Each is passed as an
    // additional parameter to the kernel. Locally-const globals are passed by
    // value. Non-const globals are passed by reference.
    Individual,
  };

  using DeviceToHostMap =
      std::map<llvm::GlobalVariable *, llvm::GlobalVariable *>;

  using GlobalVariables = std::vector<GlobalVariable*>;

private:
  Mode mode;
  Module& DeviceModule;

  // Maps the device globals to the corresponding host globals. All global
  // variables from the device module must have an entry in this map.
  DeviceToHostMap deviceToHostMap;

public:
  friend class LocalizeGlobalsImpl;

public:
  // This constructor is used when the host and device modules are the same
  // and all named globals in the module are to be localized.
  LocalizeGlobals(LocalizeGlobals::Mode mode, Module& M);

  // This constructor is used when all named globals in the host module need to
  // be localized in the device module.
  LocalizeGlobals(LocalizeGlobals::Mode mode, Module &DeviceModule,
                  Module &HostModule);

  // This constructor is used when only a subset of the variables in the
  // device module need to be localized. This is useful in cases where the host
  // and device modules are the same but only the global variables used in some
  // function(s) need to be localized.
  LocalizeGlobals(LocalizeGlobals::Mode mode, Module &DeviceModule,
                  const GlobalVariables &HostGlobals);

  bool localizeGlobalsInDeviceFunction(Function& DeviceFunction,
                                       Module& HostModule);
};

} // namespace llvm

#endif // TapirLOCALIZE_GLOBALS_H
