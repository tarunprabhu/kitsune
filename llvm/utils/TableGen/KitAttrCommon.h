//===- KitAttrCommon.h - Common code for Kitsune's attributes -----*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities shared by Kitsune's attribute emitters.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_KIT_ATTR_COMMON_H
#define LLVM_TABLEGEN_KIT_ATTR_COMMON_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class Record;
class RecordKeeper;

} // namespace llvm

/// Get the base name of the attribute. The table below provides some examples
/// of how the name of the attribute will be translated to the base name:
///
///    Target               target
///    Grainsize            grainsize
///    PerfectDepth         perfect.depth
///    DeviceModuleFlags    device.module.flags
///    Dim3                 dim3
///
std::string getIRBaseName(const llvm::Record &attr);

/// Get the names of the attribute as it will appear in LLVM-IR. This is like
/// \ref getIRBaseName, but with the given prefix.
std::string getIRName(llvm::StringRef prefix, const llvm::Record &attr);

/// Check if the attribute can be added to Tapir constructs only. This requires
/// the presence of a field whose value is one of 'TapirOnly', 'NormalOnly', or
/// 'TapirOrNormal'.
bool isTapirOnly(const llvm::Record &attr);

/// Get value type. Get the type of the value in the record. This assumes that
/// a field named "ValueType" exists in the record that is an instance of class
/// "Type". The "Type" class a field named "Name". The value of this field is
/// returned.
llvm::StringRef getTypeName(const llvm::Record &attr);

#endif // LLVM_TABLEGEN_KIT_ATTR_COMMON_H
