//===- TTUtils.h - Utilities closely related to tapir targets --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Miscellaneous utilities for tapir targets and TTID's.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_TT_UTILS_H
#define KITSUNE_CORE_TT_UTILS_H

#include "kitsune/Core/SpawnStrategy.h"
#include "kitsune/Core/TTID.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

class LLVMContext;
class TTOptions;

using OwnedModule = std::unique_ptr<Module>;

/// Get the "support" module for the given tapir target.
///
/// The "support" module is typically obtained by parsing the contents of one
/// or more bitcode files. The purpose of this module depends on the tapir
/// target. For instance, for the 'cuda' and 'hip' tapir targets, this module
/// is obtained from one or more libdevice bitcode files. On the other hand, for
/// the 'opencilk' tapir target, the support module is obtained from a bitcode
/// file that is used to interface with the actual runtime. Not all tapir
/// targets require a support module. Calling this function with the TTID of
/// such a tapir target will result in an error.
///
/// More than one module may have to be linked in order to produce the result.
/// The files to be parsed are obtained from the given tapir target options
/// object. Calling this function multiple times can be expensive, so callers
/// should cache the returned module when possible.
///
/// \param tt The TTID for which to construct the support module
/// \param tto The tapir target options object used to determine which files to
///            parse in order to obtain the support module
/// \param ctx The LLVM context. This should be the same context as is used by
///            the "main" module being compiled
Expected<OwnedModule> getSupportModule(TTID tt, const TTOptions &tto,
                                       LLVMContext &ctx);

/// Get the libdevice module for the given tapir target. The tapir target must
/// be one either 'cuda' or 'hip' since those are the only tapir targets that
/// use a libdevice module. This internally just calls \ref getSupportModule.
/// See \ref getSupportModule for more information.
Expected<OwnedModule> getLibDeviceModule(TTID tt, const TTOptions &tto,
                                         LLVMContext &ctx);

/// Get the spawn strategy for the given tapir target. At the time of writing,
/// the spawn strategy is still required in some places, though its actual
/// utility is questionable. With the exception of 'opencilk', none of the other
/// tapir targets could even, in principle, use this in any meaningful way. As
/// a result, the spawn strategy can be inferred from the tapir target. This
/// function provides that mapping.
TapirSpawnStrategy getSpawnStrategyFor(TTID tt);

/// Has the given tapir target been enabled in this build.
bool isEnabledTT(TTID tt);

/// Does the tapir target generate code that will run on a GPU. Note that this
/// will always return false for the 'custom' tapir target.
bool isGPUTT(TTID tt);

/// Does the tapir target generate code that will run on a CPU. This will always
/// return `false` for the custom tapir target. This will return `true` for the
/// serial tapir target as well as those that generate multithreaded code.
bool isCPUTT(TTID tt);

/// Check if the given tapir target generates embedded bitcode.
bool generatesEmbBC(TTID tt);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_TT_UTILS_H
