//===- GVAttrDocs.td - Docs for Kitsune's global var attrs -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Documentation for Kitsune-specific global variable attributes. These are used
// to generate the HTML documentation.
//
//===----------------------------------------------------------------------===//

def KitBitCodeDoc : KitDocAttr {
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

def KitDeviceCodeDoc : KitDocAttr {
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

def KitKernelPropertiesDoc : KitDocAttr {
  let Documentation = [{
The global variable contains properties of a kitsune "kernel function". The
value of the attribute is the name of the kernel function. The function itself
will be defined in an embedded bitcode module.
}];
}
