//===- FuncAttrDocs.td - Docs for Kitsune's function attrs -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Documentation for Kitsune-specific function attributes. These are used to
// generate the HTML documentation.
//
//===----------------------------------------------------------------------===//

def KitDeviceDoc : KitDocAttr {
  let Documentation = [{
The function is a "device function". This is a function in the device module
that is reachable from at least one kernel function. This attribute must **not**
be applied to "library" functions such as those provided by the libdevice
bitcode files.
}];
}

def KitEarlyAnnotatePassDoc : KitDocAttr {
  let Documentation = [{
The presence of this attribute indicates that the `kit-annotate-early` pass has
been run on the function.
}];
}

def KitKernelDoc : KitDocAttr {
  let Documentation = [{
The function is a "kernel function". This is a function in the device module
that was obtained by outlining a tapir loop. With GPU-centric tapir targets,
this is the entry point into the GPU and will be launched onto the GPU by
Kitsune's runtime.
}];
}

def KitPreLowerAnnotatePassDoc : KitDocAttr {
  let Documentation = [{
The presence of this attribute indicates that the `kit-annotate-prelower`
pass has been run on the function.
}];
}

def KitSerializePassDoc : KitDocAttr {
  let Documentation = [{
The presence of this attribute indicates that the `kit-serialize` pass has been
run on the function.
}];
}
