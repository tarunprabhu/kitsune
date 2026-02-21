# LLVM Attribute Reference

These are Kitsune-specific LLVM attributes. These are either emitted by
Kitsune's drivers during LLVM-IR generation, or introduced by Kitsune's LLVM
passes. These are "true" LLVM attributes, in the sense that they each have an
entry in LLVM's `Attribute::AttrKind` enum. In addition to these, Kitsune also
supports a number of "loop attributes". These are not true attributes since they
only appear in the metadata attached to an LLVM loop. Nevertheless, Kitsune
provides a consistent way of defining and accessing these. They are
documented in a [dedicated page](KitLoopAttrsDoc).

## Function Attributes

These are attributes that only apply to functions.

(llvm-attr-kit-device)=
**kit_device**
: This attribute indicates that a function is a
  [device function](glossary-device-function), that is, it only runs on an
  accelerator and can only be called by code that is already running on an
  accelerator. This attribute may only be applied to functions in a
  [device module](glossary-device-module). Thee functions and must be reachable
  from at least one [kernel function](glossary-kernel-function).

  This attribute must _*not*_ be applied to "library" functions i.e.
  functions whose definitions will be provided by
  [libdevice bitcode files](glossary-libdevice-bitcode-file).

(llvm-attr-kit-kernel)=
**kit_kernel**
: This attribute indicates that a function is a
  [kernel function](glossary-kernel-function), that is, it only runs on an
  accelerator and is "launched" (as opposed to "called") by code running on the
  [host](glossary-host). This attribute may only be applied to functions in a
  [device module](glossary-device-module).

## Global Attributes

These attributes only apply to global variables.

(llvm-attr-kit-bc)=
**kit_bc**
: This attribute indicates that the initializer of the global variable consists
  of [embedded bitcode](glossary-embedded-bitcode). If this attribute is present,
  the [kit\_tt](llvm-attr-kit-tt) attribute must also be present.

(llvm-attr-kit-fb)=
**kit_fb**
: This attribute indicates that the initializer of the global variable consists
  of [device code](glossary-device-code). If this attribute is present, the
  [kit\_tt](llvm-attr-kit-tt) attribute must also be present.

(llvm-attr-kit-kernel-props)=
**kit_kernel_props**(`name: string`)
: This attribute indicates that the initializer of the global variable contains
  the serialized "properties" of a [kernel function](glossary-kernel-function).
  The value of this attribute is the name of the kernel function.

(llvm-attr-kit-tt)=
**kit_tt**(`N: int`)
: This attribute indicates that the global variable was created by a
  [tapir target](glossary-tapir-target). The value of the attribute is
  the integer representation of the tapir target that generated it. This
  attribute must be accompanied by exactly one of
  [kit\_bc](llvm-attr-kit-bc) or [kit\_fb](llvm-attr-kit-fb).

## Loop Attributes

These are documented [here](KitLoopAttrsDoc).

## Parameter Attributes

These attributes only apply to function parameters.

```{note}
Currently, there are no Kitsune-specific parameter attributes
```

## Type Attributes

These attributes only apply to LLVM types.

```{note}
Currently, there are no Kitsune-specific type attributes.
```
