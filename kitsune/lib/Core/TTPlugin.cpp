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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

Expected<TTPlugin> TTPlugin::load(StringRef dsoPath) {
  std::string err;
  auto dylib = sys::DynamicLibrary::getPermanentLibrary(dsoPath.data(), &err);
  if (!dylib.isValid())
    return make_error<StringError>(
        join_items("", "Could not load library '", dsoPath, "': ", err),
        inconvertibleErrorCode());

  TTPlugin plugin{dsoPath.data(), dylib};

  // llvmGetPassPluginInfo should be resolved to the definition from the plugin
  // we are currently loading.
  intptr_t getDetailsFn =
      (intptr_t)dylib.getAddressOfSymbol("llvmGetTTPluginInfo");

  if (!getDetailsFn)
    // If the symbol isn't found, this is probably a legacy plugin, which is an
    // error
    return make_error<StringError>(
        join_items("", "Plugin entry point not found in '", dsoPath, "'"),
        inconvertibleErrorCode());

  plugin.info =
      reinterpret_cast<decltype(llvmGetTTPluginInfo) *>(getDetailsFn)();

  if (plugin.getAPIVersion() != LLVM_TTPLUGIN_API_VERSION)
    return make_error<StringError>(
        llvm::join_items(
            "", "Wrong API version on plugin '", dsoPath, "'. Got version ",
            std::to_string(plugin.getAPIVersion()), ", supported version is ",
            std::to_string(LLVM_TTPLUGIN_API_VERSION)),
        inconvertibleErrorCode());

  if (!plugin.info.makeTapirTarget)
    return createStringError(join_items(
        "", "Missing constructor callback in plugin '", dsoPath, "'"));

  if (!plugin.info.getCompilerOptions)
    return createStringError(join_items(
        "", "Missing compiler options callback in plugin '", dsoPath, "'"));

  if (!plugin.info.getLinkerOptions)
    return createStringError(join_items(
        "", "Missing linker options callback in plugin '", dsoPath, "'"));

  return plugin;
}
