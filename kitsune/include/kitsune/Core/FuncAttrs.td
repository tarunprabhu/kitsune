//===- FunctionAttrs.td - Defines attributes for function --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Kitsune-specific attributes that can be added to functions.
//
//===----------------------------------------------------------------------===//

include "kitsune/Core/AttrCommon.td"
include "kitsune/Core/FuncAttrDocs.td"

// Base class for all Kitsune-specific function attributes.
//
// The names of the concrete instances of this class will be the name of the
// attribute as it appears in the FuncAttrKind enum. This must be in Pascal
// case, and consist of letters and digits only. The name of the attribute as it
// appears in LLVM-IR metadata is derived from this.
//
//     .-----------------------------------------.
//     | Name          | Metadata string         |
//     |-----------------------------------------|
//     | Kernel        | kit.func.kernel         |
//     | KernelProps   | kit.func.kernel.props   |
//     | Reduce3       | kit.func.reduce3        |
//     '-----------------------------------------'
//
class FuncAttr<Type ValueType = VoidTy> : AttrBase<ValueType> { }

//------------------------------------------------------------------------------

// Flag indicating that the function is a "device function".
def Device : FuncAttr<> {
  let HasCustomVerifier = true;
  let Documentation = KitDeviceDoc;
}

// Flag to indicate that the kit-annotate-early pass has been run on the
// function.
def EarlyAnnotatePass : FuncAttr<> {
  let Documentation = KitEarlyAnnotatePassDoc;
}

// Flag indicating that the function is a "kernel function".
def Kernel : FuncAttr<> {
  let HasCustomVerifier = true;
  let Documentation = KitKernelDoc;
}

// Flag indicating that the kit-annotate-prelower pass has been run on the
// function.
def PreLowerAnnotatePass : FuncAttr<> {
  let Documentation = KitPreLowerAnnotatePassDoc;
}

// Flag indicating that the kit-serialize pass has been run on the function.
def SerializePass : FuncAttr<> {
  let Documentation = KitSerializePassDoc;
}
