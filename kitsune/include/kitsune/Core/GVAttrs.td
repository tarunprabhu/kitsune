//===- GVAttrs.td - Defines global variable attributes -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Kitsune-specific attributes that can be added to global variables.
//
//===----------------------------------------------------------------------===//

include "kitsune/Core/AttrCommon.td"
include "kitsune/Core/GVAttrDocs.td"

// Base class for all Kitsune-specific global variable attributes.
//
// The names of the concrete instances of this class will be the name of the
// attribute as it appears in the GlobalVariableAttrKind enum. This must be in
// Pascal case, and consist of letters and digits only. The name of the
// attribute as it appears in LLVM-IR metadata is derived from this.
//
//     .-----------------------------------------.
//     | Name          | Metadata string         |
//     |-----------------------------------------|
//     | Bitcode       | kit.gv.bitcode          |
//     | DeviceCode    | kit.gv.device.code      |
//     | Dim3          | kit.gv.dim3             |
//     '-----------------------------------------'
//
class GVAttr<Type ValueType = VoidTy> : AttrBase<ValueType> { }

//------------------------------------------------------------------------------

// The global variable contains an LLVM module serialized as Base64-encoded
// LLVM bitcode. The attribute value is the ID of the tapir target that
// created this global.
def BitCode : GVAttr<TTIDTy> {
  let HasCustomVerifier = true;
  let Documentation = KitBitCodeDoc;
}

// The global variable consists of compiled device code (usually for a GPU).
// The attribute value is the ID of the tapir target that created this global.
def DeviceCode : GVAttr<TTIDTy> {
  let HasCustomVerifier = true;
  let Documentation = KitDeviceCodeDoc;
}

// The global variable contains the properties of a GPU kernel - typically one
// generated from a tapir loop nest. The attribute value is a tuple consisting
// of the ID of the tapir target that generated the global and the name of the
// kernel as it would appear in the device code.
def KernelPropertiesTy
    : TupleType<[TupleElement<TTIDTy, "TTID">,
                 TupleElement<StrTy, "Name">]> {}
def KernelProperties : GVAttr<KernelPropertiesTy> {
  let HasCustomVerifier = true;
  let Documentation = KitKernelPropertiesDoc;
}
