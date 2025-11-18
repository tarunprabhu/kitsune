//===- TTPlugin.h - Public interface for tapir target plugins --*- C++ -*--===//
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

#ifndef KITSUNE_CORE_TT_PLUGIN_H
#define KITSUNE_CORE_TT_PLUGIN_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"

namespace llvm {

class Module;
class TapirTarget;
class TTOptions;

/// \macro TT_PLUGIN_API_VERSION
/// Identifies the API version understood by this plugin.
///
/// When a plugin is loaded, the driver will check it's supported plugin version
/// against that of the plugin. A mismatch is an error. The supported version
/// will be incremented for ABI-breaking changes to the \c TTPluginInfo
/// struct, i.e. when callbacks are added, removed, or reordered.
#define LLVM_TTPLUGIN_API_VERSION 1

extern "C" {

/// Information about the plugin required to load its passes
///
/// This struct defines the core interface for pass plugins and is supposed to
/// be filled out by plugin implementors. LLVM-side users of a plugin are
/// expected to use the \c PassPlugin class below to interface with it.
struct TTPluginInfo {
  /// The API version understood by this plugin, usually \c
  /// LLVM_TTPLUGIN_API_VERSION
  uint32_t apiVersion;

  /// A meaningful name of the plugin.
  const char *pluginName;

  /// The version of the plugin.
  const char *pluginVersion;

  /// The callback to construct the tapir target object that will be used. The
  /// caller will own the returned object.
  TapirTarget *(*makeTapirTarget)(Module &, const TTOptions &);

  /// Callback to get any options that should always be added to the compiler
  /// (cc1/fc1) when using the plugin. If no additional compiler options are
  /// required, return an empty string.
  SmallVector<std::string, 4> (*getCompilerOptions)();

  /// Callback to get any options that should always be added to the linker when
  /// using the plugin. If no additional linker options are required, return
  /// an empty string.
  SmallVector<std::string, 4> (*getLinkerOptions)();
};

} // extern "C"

/// A loaded tapir target plugin.
///
/// An instance of this class wraps a loaded tapir target plugin and gives
/// access to its interface defined by the \c TTPluginInfo it exposes.
class TTPlugin {
public:
  using ExtraArgsList = SmallVector<std::string, 4>;

private:
  /// The path to the dynamic shared object that is the plugin
  std::string dsoPath;

  /// LLVM's wrapper around the loaded dynamic library
  sys::DynamicLibrary dylib;

  /// The plugin information struct
  TTPluginInfo info;

private:
  TTPlugin(const std::string &dsoPath, const sys::DynamicLibrary &dylib)
      : dsoPath(dsoPath), dylib(dylib), info() {}

public:
  /// Attempts to load a tapir target plugin from a given file.
  ///
  /// \returns Returns an error if either the library cannot be found or loaded,
  /// there is no public entry point, or the plugin implements the wrong API
  /// version.
  LLVM_ABI static Expected<TTPlugin> load(StringRef dsoPath);

  /// Get the name loaded plugin file.
  StringRef getFile() const { return dsoPath; }

  /// Get the plugin name
  StringRef getName() const { return info.pluginName; }

  /// Get the plugin version
  StringRef getVersion() const { return info.pluginVersion; }

  /// Get the plugin API version
  uint32_t getAPIVersion() const { return info.apiVersion; }

  /// Construct a tapir target object. The caller will own the constructed
  /// object.
  TapirTarget *makeTapirTarget(Module &hostM, const TTOptions &tto) const {
    return info.makeTapirTarget(hostM, tto);
  }

  /// Return any options that must always be added to the compiler (cc1/fc1)
  /// when using this plugin.
  ExtraArgsList getCompilerOptions() const { return info.getCompilerOptions(); }

  /// Return any options that must always be added to the linker when using this
  /// plugin.
  ExtraArgsList getLinkerOptions() const { return info.getLinkerOptions(); }
};

} // namespace llvm

/// The public entry point for a tapir target plugin.
///
/// When a plugin is loaded by the driver, it will call this entry point to
/// obtain information about this plugin. This function needs to be implemented
/// by the plugin, see the example below:
///
/// ```
/// extern "C" ::llvm::TTPluginInfo LLVM_ATTRIBUTE_WEAK
/// llvmGetTTPluginInfo() {
///   return {
///     LLVM_TTPLUGIN_API_VERSION, "MyPlugin", "v0.1",
///     [](Module &hostM, const TTOptions &tto) {
///       // return a new tapir target
///     },
///     [] { ... // return a (possibly empty) array of compiler options },
///     [] { ... // return a (possibly empty) array of linker options }
///   };
/// }
/// ```
extern "C" ::llvm::TTPluginInfo LLVM_ATTRIBUTE_WEAK llvmGetTTPluginInfo();

#endif // KITSUNE_CORE_TT_PLUGIN_H
