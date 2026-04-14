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

def BitCode : GVAttr<TTIDTy> {
  let HasCustomVerifier = true;
  let Documentation = [{
Indicates that the initializer of the global variable contains serialized LLVM
bitcode. This bitcode will usually be compiled to machine code for a device -
usually a GPU. This variable will have been added by a tapir target. The value
of this attribute is the ID of the tapir target that generated the bitcode in
the initializer.

The module that is parsed from the contents of this initializer
is referred to as an "embedded module" or "embedded bitcode". Passes that
operate on this module are embedded bitcode passes.
}];
}

def DeviceCode : GVAttr<TTIDTy> {
  let HasCustomVerifier = true;
  let Documentation = [{
Indicates that the initializer of the global variable contains machine code for
an accelerator, usually a GPU. This variable will have been added by a tapir
target. The value of this attribute is the ID of the tapir target that generated
the machine code in the initializer.

The initializer of such globals is usually only set when the codegen passes
are run. Therefore, most middle-end passes are likely to see a global with a
zero-initializer.
}];
}

def KernelPropertiesTy
    : TupleType<[TupleElement<TTIDTy, "TTID">,
                 TupleElement<StrTy, "Name">]> {}
def KernelProperties : GVAttr<KernelPropertiesTy> {
  let HasCustomVerifier = true;
  let Documentation = [{
The global variable contains properties of a kitsune "kernel function". The
value of the attribute is the name of the kernel function. The function itself
will be defined in an embedded bitcode module.
}];
}
