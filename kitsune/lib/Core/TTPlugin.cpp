//===- TTPlugin.cpp - Public interface for tapir target plugins -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines the public entry point for tapir target plugins
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TTPlugin.h"
#include "kitsune/Core/Diagnostics.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

Expected<TTPlugin> TTPlugin::load(StringRef dsoPath) {
  std::string err;
  auto dylib = sys::DynamicLibrary::getPermanentLibrary(dsoPath.data(), &err);
  if (!dylib.isValid())
    return createDiagError(DiagID::ErrTTPluginLoad, dsoPath, err);

  TTPlugin plugin{dsoPath.data(), dylib};

  // llvmGetPassPluginInfo should be resolved to the definition from the plugin
  // we are currently loading.
  intptr_t getDetailsFn =
      (intptr_t)dylib.getAddressOfSymbol("llvmGetTTPluginInfo");

  if (!getDetailsFn)
    // If the symbol isn't found, this is probably a legacy plugin, which is an
    // error
    return createDiagError(DiagID::ErrTTPluginEntryPoint, dsoPath);

  plugin.info =
      reinterpret_cast<decltype(llvmGetTTPluginInfo) *>(getDetailsFn)();

  if (plugin.getAPIVersion() != LLVM_TTPLUGIN_API_VERSION)
    return createDiagError(DiagID::ErrTTPluginAPIVersion, dsoPath,
                           plugin.getAPIVersion(), LLVM_TTPLUGIN_API_VERSION);

  if (!plugin.info.makeTapirTarget)
    return createDiagError(DiagID::ErrTTPluginCBConstructor, dsoPath);

  if (!plugin.info.getCompilerOptions)
    return createDiagError(DiagID::ErrTTPluginCBCompilerOpts, dsoPath);

  if (!plugin.info.getLinkerOptions)
    return createDiagError(DiagID::ErrTTPluginCBLinkerOpts, dsoPath);

  return plugin;
}
