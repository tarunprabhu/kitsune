//===- InstAttrs.td - Defines instruction attributes -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Kitsune-specific attributes that can be added to instructions.
//
//===----------------------------------------------------------------------===//

include "kitsune/Core/AttrCommon.td"
include "kitsune/Core/InstAttrDocs.td"

// Base class for all Kitsune-specific instruction attributes. For now, the
// attributes are not constrained to be applied to only certain instruction
// kinds, but that might change in the future.
//
// The names of the concrete instances of this class will be the name of the
// attribute as it appears in the InstAttrKind enum. This must be in Pascal
// case, and consist of letters and digits only. The name of the attribute as it
// appears in LLVM-IR metadata is derived from this.
//
//     .-----------------------------------------.
//     | Name         | Metadata string          |
//     |-----------------------------------------|
//     | Safe         | kit.inst.safe            |
//     | ImplicitSync | kit.inst.tapir.loop      |
//     | Reduce3      | kit.inst.reduce3         |
//     '-----------------------------------------'
//
class InstAttr<Type ValueType = VoidTy> : AttrBase<ValueType> { }

//------------------------------------------------------------------------------

// The tapir target for a detach, reattach, or sync, instruction.
def Target : InstAttr<TTIDTy> {
  let HasCustomVerifier = true;
  let Documentation = KitInstTargetDoc;
}
