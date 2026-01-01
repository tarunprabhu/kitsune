# Command-Line Options

This document describes the considerations that informed the design and
organization of Kitsune's command-line options. The document on
[adding a command-line option](AddingCommandLineOption.md) goes into much
greater detail, but that document is primarily concerned with the actual
implementation in Kitsune, and the [drivers](glossary-driver) in particular.
This also describes the factors that should be taken into consideration when
adding a command-line option to Kitsune.


## General Rules

1. Command-line options that pertain to [tapir targets](glossary-tapir-target)
   and code generation should be available to both the drivers and
   LLVM tools. For instance, `--tapir-cuda-arch`, which specifies the GPU
   architecture for which to generate code when using the
   [cuda](tapir-targets-cuda) tapir target, are available in `kitcc`, `kit++`,
   and `kitfc`, as well as [opt](https://llvm.org/CommandGuide/opt.html). This
   is to make it easier to use the drivers with high-level
   [source](glossary-source-language) code, and [LLVM tools](LLVMTools.md) with
   [LLVM-IR](glossary-llvm-ir). The spelling and default values of such options
   should be identical in both cases, as should the help message, if possible.
   As of Jan 2026, it is not possible to share command-line options
   automatically between the drivers and the LLVM tools, so the option must be
   duplicated.

2) Such options should be spelled with a double-hyphen at the start, not a
   single hyphen. For instance, `--tapir-verbose` should be used, not
   `-tapir-verbose`. Newer options that are being added to
   [Clang](https://clang.llvm.org) and [Flang](https://flang.llvm.org) follow
   this convention, so we do so as well. Most existing LLVM options already
   use double-hyphens. This ensures greater conformity with modern practices.

3. When adding a command-line option that is only relevant to the drivers,
   avoid using prefixes such as `-f`. Kitsune does support specifying the tapir
   target using `-ftapir`, but this has been deprecated in favor of `--tapir`.
   The exception to this is when adding an option that should be exposed to
   `clang` and/or `flang`. Such options would be different from Kitsune-specific
   options as most of those may only be used with Kitsune's drivers.

4) Avoid adding command-line options to [LLD](https://lld.llvm.org/) when
   possible, preferring instead to use the LLVM options that are already
   available. This is described in greater detail [here](lto-implementation).

5. Command-line options that pertain to tapir targets in general should start
   with `--tapir-`. Command-line options specific to a given tapir target
   should start with `--tapir-<tapir target>-`. For instance, `--tapir-cuda-`,
   `--tapir-opencilk-` and so on.


## Driver Option or LLVM Option

When adding a command-line option specific to a tapir target, we need to decide
whether the option should be exposed to the driver, or if it should be
"private" to the tapir target i.e. it is only available in LLVM. It is still
possible to use this option with the driver, by using `-mllvm`. For instance,
a hypothetical `--tapir-private-option` defined as an LLVM
[command-line option](https://llvm.org/docs/CommandLine.html) can be used as
shown here.

```shell
kitcc -mllvm --tapir-private-option ...
```

It is clear that such an option is not particularly ergonomic. Some questions
that could be asked when deciding whether to make the option a first-class
driver option are:

- **Is the option likely to make Kitsune easier to use/debug?**

  For instance, the `--tapir-verbose` option may not be used frequently by
  end-users, but may be used often by Kitsune developers, especially when
  debugging. In such cases, it is reasonable to make this option available in
  the driver to avoid having to constantly type `-mllvm --tapir-verbose`.

- **Is the default value of the option likely to result in mis-compilation or
  runtime failures on certain platforms?**

  An example of this is the `--tapir-hip-xnack` option used by the
  [hip](tapir-targets-hip) tapir target. The value of this option can be one
  of `on`, `off`, or `any`. LLVM's
  [documentation](https://llvm.org/docs/AMDGPUUsage.html#target-features)
  suggests that, if xnack support is enabled in the device, code compiled
  without xnack support - in our case with `--tapir-hip-xnack=off` - can raise
  a runtime error if a page fault occurs. Since Kitsune cannot reliably
  determine if xnack support is available, the onus is on the user to use the
  option with the correct value. In such cases, the option must be exposed in
  the driver.

- **Is the option likely to only be useful to Kitsune developers?**

  This is the case when an option controls features of the hardware or runtime
  system that are not well-documented. Such options are generally only useful
  to Kitsune developers seeking to understand if there is a performance
  advantage to be gained by using/avoiding the feature. Incorrect use of such
  options may result in mis-compilation or incorrect execution. Such options
  should not generally be exposed to the driver. Sophisticated users who
  understand the tradeoffs can access such options using `-mllvm`. A complete
  list of LLVM options can be obtained using LLVM's `opt` tool under the
  "Kitsune Options" category.

  ```shell
  opt -help
  ```

These also apply when adding a command-line option that is not specific to a
tapir target.


## Option Validation

Currently, not all Kitsune-specific LLVM options are validated. Some options
cannot be easily validated. For instance, `--tapir-cuda-arch`, whose value is
the architecture of an NVIDIA GPU. The list of architectures is not available in
LLVM - obtaining such a list would require duplicating code from clang. Since
the LLVM options are are most likely to be used by Kitsune developers or
sophisticated users, this is a reasonable trade-off. However, where possible,
these options should be validated.

Driver options **must** be validated.
