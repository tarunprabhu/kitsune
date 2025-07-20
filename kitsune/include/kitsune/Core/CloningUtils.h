//===- CloningUtils.h - Utilities to aid in cloning entities ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to support cloning code from host to embedded modules.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_CLONING_UTILS_H
#define KITSUNE_CORE_CLONING_UTILS_H

#include "kitsune/Core/Tapir.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {

class Module;
class ReachableGlobals;

/// Clone the reachable GlobalValue's into the given device module. This
/// function is specially geared towards cloning the globals into a device
/// module. It does not necessarily retain all properties of the source globals
/// exactly. For instance, depending on the given tapir target id, the address
/// space of the cloned global may not be the same as that of the host.
/// Similarly, the unnamed_addr flag is set on the clone, even if it is not set
/// on the source.
void cloneGlobalValuesInto(const ReachableGlobals &globals, TTID tt,
                           Module &devM, ValueToValueMapTy &vmap);

/// Clone the reachable GlobalValue's into the given device module. This entry
/// point is useful if we don't need to populate a ValueMap between the source
/// and cloned values.
void cloneGlobalValuesInto(const ReachableGlobals &globals, TTID tt,
                           Module &devM);

} // namespace llvm

#endif // KITSUNE_CORE_CLONING_UTILS_H
