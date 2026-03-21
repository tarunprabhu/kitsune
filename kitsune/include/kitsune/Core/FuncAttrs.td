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

def Device : FuncAttr<> {
  let Documentation = [{
The function is a "device function". This is a function in the device module
that is reachable from at least one kernel function. This attribute must **not**
be applied to "library" functions such as those provided by the libdevice
bitcode files.
}];
}

def Kernel : FuncAttr<> {
  let Documentation = [{
The function is a "kernel function". This is a function in the device module
that was obtained by outlining a tapir loop. With GPU-centric tapir targets,
this is the entry point into the GPU and will be launched onto the GPU by
Kitsune's runtime.
}];
}
