//===- ModuleUtils.h - Utilities for LLVM modules --------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM Module's.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_MODULE_UTILS_H
#define KITSUNE_CORE_MODULE_UTILS_H

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace llvm {

class LLVMContext;
class Module;
class NamedMDNode;

/// \addtogroup kitsune
/// @{

/// Get the LLVM context from a module. This is useful when generating code from
/// tablegen macros.
LLVMContext &getContext(Module &m);
LLVMContext &getContext(const Module &m);

/// Add a the kit.module.device.module attribute containing information about
/// the module into itself. This should only be called with a module generated
/// by a tapir target (currently, these are only the GPU-centric tapir targets).
///
/// The original reason for adding this metadata is to ensure that the module
/// name can be restored when it is deserialized from the global variable into
/// which it is embedded. Without this, the name of the module is lost. While it
/// is not wrong if the module does not have an identifier, it is less helpful
/// if it does not. Since then, it has been updated to include some other
/// information as well.
///
/// Currently, this named metadata node looks like this:
///
///   kit.device.module.flags = !{!0, !1}
///
///   !0 = !{!i32 [[TTID]]}
///   !1 = !{!"[[MODULE_NAME]]"}
///
/// where
///
///   [[TTID]]
///
///       is the integer representation of the tapir target that generated
///
///   [[MODULE_NAME]]
///
///       is the module identifier as obtained by calling Module::getName()
///
/// \param m The module
/// \param tt The tapir target that created the device module \p m
void addDeviceModuleFlagsAttr(Module &m, TTID tt);

/// Clone LLVM's module flags metadata from the host into the device module.
/// This will not clone every operand of the metadata, but only a subset that
/// are guaranteed to be "safe". Some of the operands in that metadata node are
/// set based on the command line options. However, in our case, the IR that is
/// generated is assumed to run on the host. As a result, it may contain options
/// that are only relevant on the host, or, in the worst case, options that are
/// actually incorrect when set on a module to be compiled for a GPU.
///
/// FIXME: We should try and determine if there is a way to construct a filter
/// that will copy *all* operands *except* those that are known to be unsafe.
void cloneModuleFlagsMetadataInto(Module &devM, const Module &hostM);

/// Clone LLVM's ident metadata from the host into the device module.
void cloneIdentMetadataInto(Module &devM, const Module &hostM);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_MODULE_UTILS_H
