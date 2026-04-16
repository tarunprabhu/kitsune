//===- ArgAttrs.td - Attributes for function arguments -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Kitsune-specific attributes that can be added to functions
// argumnents.
//
//===----------------------------------------------------------------------===//

include "kitsune/Core/AttrCommon.td"
include "kitsune/Core/ArgAttrDocs.td"

// Base class for all Kitsune-specific argument attributes.
//
// The names of the concrete instances of this class will be the name of the
// attribute as it appears in the FuncAttrKind enum. This must be in Pascal
// case, and consist of letters and digits only. The name of the attribute as it
// appears in LLVM-IR metadata is derived from this.
//
//     .-----------------------------------------.
//     | Name          | Metadata string         |
//     |-----------------------------------------|
//     | Readonly      | kit.arg.readonly        |
//     | ReadWrite     | kit.arg.read.write      |
//     | Reduce3       | kit.arg.reduce3         |
//     '-----------------------------------------'
//
class ArgAttr<Type ValueType = VoidTy> : AttrBase<ValueType> { }

//------------------------------------------------------------------------------
