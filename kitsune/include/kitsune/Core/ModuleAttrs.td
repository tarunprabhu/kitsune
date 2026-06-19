//===- ModuleAttrs.td - Module-level attributes ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Module-level "attributes". These are really named metadata nodes in a module.
//
//===----------------------------------------------------------------------===//

include "kitsune/Core/AttrCommon.td"
include "kitsune/Core/ModuleAttrDocs.td"

// Base class for Kitsune-specific module attributes.
//
// The names of the concrete instances of this class will be the name of the
// attribute as it appears in the ModuleAttrKind enum. This must be in Pascal
// case, and consist of letters and digits only. The name of the attribute as it
// appears in LLVM-IR metadata is derived from this. The table below shows
// some examples of this.
//
//     .--------------------------------------------------.
//     | Name           | Metadata string                 |
//     |--------------------------------------------------|
//     | Device         | kit.module.loop.device          |
//     | LoopsAnnotated | kit.module.loop.loops.annotated |
//     | Bytesize       | kit.module.loop.bytesize        |
//     | Version4       | kit.module.loop.version4        |
//     '--------------------------------------------------'
//
// The attributes may have values associated with them. In most cases, a number
// of accessor functions will be generated for each attribute declared here.
// These are generally of the form
//
//     bool has<NAME>Attr(const Module &m);
//     void add<NAME>Attr(Module &m, ...);
//     void remove<Name>Attr(Module &m);
//
// Here <NAME> is the name of the concrete instance of the attribute that has
// been defined here. The add<NAME>attr function will accept 0 or more
// arguments depending on the number of values that the attribute has. For a
// "flag" attribute that does not accept a value, the signature will be:
//
//     void add<NAME>Attr(Module &m);
//
// while for one that accepts a TTID enum and a string, it will be:
//
//     void add<NAME>Attr(Module &m, TTID tt, StringRef s);
//
class ModuleAttr<Type ValueType = VoidTy> : AttrBase<ValueType> { }

//------------------------------------------------------------------------------

// Metadata to add to an embedded device module. These are similar to the
// contents of the llvm.ident metadata that is attached by clang to the "host"
// module.
def DeviceModuleFlagsTy
    : TupleType<[TupleElement<TTIDTy, "TTID">,
                 TupleElement<StrTy, "Name">]> {}
def DeviceModuleFlags : ModuleAttr<DeviceModuleFlagsTy> {
  let HasCustomVerifier = true;
  let Documentation = KitDeviceModuleFlagsDoc;
}
