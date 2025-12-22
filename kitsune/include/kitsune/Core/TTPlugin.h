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

/// \addtogroup kitsune
/// @{

class Module;
class TapirTarget;
class TTOptions;

/// Identifies the API version understood by a tapir target plugin.
///
/// When a plugin is loaded, the driver will compare the plugin version that it
/// with that of the plugin. A mismatch is an error. The supported version
/// will be incremented for ABI-breaking changes to the \ref TTPluginInfo
/// struct, i.e. when callbacks, or any static informational fields are added,
/// removed, or reordered.
#define LLVM_TTPLUGIN_API_VERSION 1

extern "C" {

/// Information about a tapir target plugin.
///
/// An instance of this is returned by the \ref llvmGetTTPluginInfo, the
/// entry-point that all tapir target plugins are required to implement.
struct TTPluginInfo {
  /// The API version understood by this plugin, usually \c
  /// LLVM_TTPLUGIN_API_VERSION
  uint32_t apiVersion;

  /// The plugin name. This is only meaningful to plugin developers and users.
  /// Kitsune will not use this for anything except, perhaps, to display it in
  /// debug messages.
  const char *pluginName;

  /// The plugin version. This is only meaningful to plugin developers and
  /// users. Kitsune will not use this for anything except, perhaps, to display
  /// it in debug messages.
  const char *pluginVersion;

  /// Callback to construct the tapir target object. The caller will take
  /// ownership of the returned object.
  TapirTarget *(*makeTapirTarget)(Module &, const TTOptions &);

  /// Callback to get any options that should always be added to the compiler
  /// (cc1/fc1) when using the plugin. May return an empty vector if no
  /// additional compiler options are required.
  SmallVector<std::string, 4> (*getCompilerOptions)();

  /// Callback to get any options that should always be added to the linker when
  /// using the plugin. May return an empty vector if no additional linker
  /// options are required.
  SmallVector<std::string, 4> (*getLinkerOptions)();
};

} // extern "C"

/// A loaded tapir target plugin.
///
/// An instance of this class wraps a loaded tapir target plugin and gives
/// access to its interface defined by the \ref TTPluginInfo it exposes.
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

/// @}

} // namespace llvm

/// The public entry point for a tapir target plugin.
///
/// When a plugin is loaded by the driver, it will call this entry point to
/// obtain information about the plugin. The plugin *must* provide a definition
/// for this function. See the example below:
///
/// ```
/// extern "C" ::llvm::TTPluginInfo LLVM_ATTRIBUTE_WEAK
/// llvmGetTTPluginInfo() {
///   return {
///     LLVM_TTPLUGIN_API_VERSION,
///     "MyPlugin",
///     "0.1",
///     [](Module &hostM, const TTOptions &tto) {
///       // return a new tapir target
///     },
///     []() {
///       // return a (possibly empty) array of compiler options
///     },
///     []() {
///       // return a (possibly empty) array of linker options
///     }
///   };
/// }
/// ```
///
/// The example above simply constructs a \ref TTPluginInfo object and returns
/// it. In principle, though, the function definition can be arbitrarily
/// complex and could be used to perform any additional setup that the plugin
/// needs.
///
/// \ingroup kitsune
extern "C" ::llvm::TTPluginInfo LLVM_ATTRIBUTE_WEAK llvmGetTTPluginInfo();

#endif // KITSUNE_CORE_TT_PLUGIN_H
