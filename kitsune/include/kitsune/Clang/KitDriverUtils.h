//=- KitDriverUtils.h - Utilities for Kitsune's command-line opts -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Helper functions for Kitsune-specific command line options.
///
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CLANG_KIT_DRIVER_UTILS_H
#define KITSUNE_CLANG_KIT_DRIVER_UTILS_H

#include "kitsune/Core/Tapir.h"
#include "clang/Driver/Driver.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace llvm {

namespace driver {
class KitOptions;
} // namespace driver

namespace opt {
class Arg;
class ArgList;
class OptTable;
} // namespace opt

} // namespace llvm

namespace clang {

namespace driver {

/// Is \p progName the name of a Kitsune frontend.
bool isKitsuneFrontend(StringRef progName);

/// Validate the Kitsune-specific options in \p args.
/// If any options are invalid, a diagnostic will be emitted. The caller will
/// determine what to do if any diagnostics were emitted.
///
/// We don't want a circular dependence between this and clang's Driver object,
/// so we pass in certain values that we would otherwise have looked up in the
/// Driver object.
void checkKitOptions(const llvm::opt::ArgList &args, bool isKitsuneFrontend,
                     bool isFlangMode, bool isUsingLTO, StringRef triple,
                     unsigned amdgpuCodeObjectVersion,
                     DiagnosticsEngine &diags);

/// Get the optimzation speedup level as an integer. This is not as
/// straightforward as it might appear since clang and flang use different
/// defaults when no optimization level is provided and we have to handle
/// flags such as -Ofast and the rather infuriating -O. We need this
/// behavior to be consistent between kitcc, kit++ and kitfc, even if that
/// behavior differs from clang and flang.
unsigned getSpeedupLevel(const llvm::opt::ArgList &args,
                         clang::DiagnosticsEngine &diags);

/// Get the optimization size level as an integer. A value of 0 indicates that
/// the code is not optimized for size. The other valid values are 1 and 2
/// corresponding to -Os and -Oz respectively where the value of 2 indicates
/// aggressive optimizations for size.
unsigned getSizeLevel(const llvm::opt::ArgList &args,
                      clang::DiagnosticsEngine &diags);

/// Parse the --tapir flag if it is present and get the name of the config file
/// of the Tapir target that was specified. If the argument of the flag is
/// invalid, or if the tapir target does not use a configuration file, this will
/// return std::nullopt.
std::optional<llvm::StringRef>
getTTConfigFileName(const llvm::opt::ArgList &args);

/// Parse the --tapir flag if it is present and return the tapir target ID. This
/// should only be called when the value of the tapir --tapir flag is valid.
std::optional<llvm::TTID> parseTTIfValid(const llvm::opt::ArgList &args);

/// Parse the Kitsune-specific command line options into a KitOptions object.
/// \param kitOpts The KitOptions object into which to parse the command
/// line options
/// \param argv0 The first argument on the command line. This is the name of the
/// executable
/// \param args The command line arguments
/// \param optTable The options table
/// \param diags The diagnostics engine
/// \returns true if parsing the options was successful, false otherwise
bool parseKitsuneArgs(llvm::driver::KitOptions &kitOpts, const char *argv0,
                      const llvm::opt::ArgList &args,
                      const llvm::opt::OptTable &optTable,
                      clang::DiagnosticsEngine &diags);

} // namespace driver

} // namespace clang

#endif // KITSUNE_CLANG_KIT_DRIVER_UTILS_H
