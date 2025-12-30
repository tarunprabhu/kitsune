//===- Attributes.td - Kitsune-specific LLVM attributes -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Kitsune-specific LLVM attributes. This file is intended to
// be included in llvm/IR/Attributes.td since it uses entities that are only
// defined there.
//
//===----------------------------------------------------------------------===//

/// The global variable was added by a tapir target and contains some other
/// attribute that indicates how to interpret the global's initializer. This
/// attribute must always occur in conjunction with another attribute.
/// Currently, the auxiliary attribute must be 'kit_bc', 'kit_fb' or
/// 'kit_kernel_props'. The value of the attribute is the integer representation
/// of the tapir target that generated the global.
def KitTT : IntAttr<"kit_tt", IntersectPreserve, [GlobalAttr]>;

/// The initializer of the global variable consists of embedded bitcode. If this
/// attribute is present, then the kit_tt attribute must also be present.
def KitBC : EnumAttr<"kit_bc", IntersectPreserve, [GlobalAttr]>;

/// The initializer of the global variable consists of device code. If this
/// attribute is present, then the kit_tt attribute must also be present.
def KitFB : EnumAttr<"kit_fb", IntersectPreserve, [GlobalAttr]>;

/// The global variable contains properties of a kitsune "kernel function". The
/// value of the attribute is the name of the kernel function.
def KitKernelProps : ComplexStrAttr<"kit_kernel_props", [GlobalAttr]>;

/// The function is a "kernel function". This is a function in the device module
/// that was obtained by outlining a tapir loop. With GPU-centric tapir targets,
/// this is the entry point into the GPU and will be launched onto the GPU by
/// kitsune's runtime.
def KitKernel : EnumAttr<"kit_kernel", IntersectPreserve, [FnAttr]>;

/// The function is a "device function". This is a function in the device module
/// that is reachable from at least one kernel function. This attribute must
/// *not* be applied to "library" functions such as those provided by the
/// libdevice bitcode files.
def KitDevice : EnumAttr<"kit_device", IntersectPreserve, [FnAttr]>;
