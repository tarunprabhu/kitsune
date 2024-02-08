//===- AddrSpace.h - Support for Kitsune's address spaces -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities for Kitsune's address spaces.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_ADDRSPACE_H
#define KITSUNE_CORE_ADDRSPACE_H

#include <cstdint>

namespace llvm {

class Module;

/// \addtogroup kitsune
/// @{

/// Address spaces used by Kitsune. The values of each address space must be
/// chosen carefully to avoid conflicting with the values used by other
/// languages/targets supported by LLVM (even if those are not necessarily
/// supported by Kitsune).
///
/// NOTE: LLVM has a maximum of 24-bits for the address space, clang has 22 (23
/// usually, but we stole a bit in Kitsune for the mobile attribute). Just in
/// case, don't use more than 8 bits for Kitsune's address spaces.
///
enum KitAS : uint8_t {
  /// The default address space for all of LLVM.
  Default = 0,

  /// Address space for mobile buffers. These are regions of memory whose
  /// contents may migrate between CPU and GPU. These are typically allocated
  /// using a UVM allocator, but they need not be. The alternative is for the
  /// compiler to explicitly generate memory copy calls to and from host and
  /// device.
  Mobile = 67,
};

static_assert(sizeof(KitAS) <= 1 &&
              "Kitsune's address spaces must occupy no more than 8 bits");

// Strip kitsune-specific address spaces from all pointers in the module. These
// values are moved into the default address space. Return true if at least one
// pointer in any of Kitsune's address spaces was found in the module and
// stripped, false if the module remains unchanged.
bool stripKitAddrSpaces(Module &m);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_ADDRSPACE_H
