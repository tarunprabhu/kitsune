# Testing Strategy

This document describes how Kitsune's tests are organized and how various
parts of Kitsune should be tested.

## Overview

Tests for Kitsune are split into two categories, [core](#core-tests) tests and
[end-to-end](#end-to-end-tests) tests.

## Core Tests

The **core** tests are found present in the main Kitsune repo. These tests are
fine-grained and test individual parts of Kitsune in isolation. These tests
may include the following:

- Tests that check that command line options are correctly validated, then
  passed from the driver to the language-specific compilers such as `cc1` or
  `fc1`.

- Tests that the correct semantic error/warning is emitted for high-level
  source code that uses Kitsune's language extensions.

- Tests that Kitsune's language extensions are correctly lowered to
  MLIR/LLVM-IR.

- Tests of individual Kitsune-specific LLVM passes.

- Tests that Kitsune's intrinsics are lowered to the correct runtime calls

In general, these tests should be as small as possible and should have a
minimal number of "dependencies". For example, checks for correct handling of
command-line options should not (intentionally) require that lowering to
LLVM-IR is also implemented. Similarly, tests of a specific LLVM pass should
not require other passes to be run at test-time. It is not always possible
to satisfy such constraints, but one should make a concerted effort to do so. If
a test requires more support from other parts of the compiler, errors in
unrelated areas of the compiler may manifest as spurious test failures [^2].
Some tips for writing such tests are presented here. These are not intended to
be comprehensive. Feel free to use the existing tests both Kitsune-specific and
those in the broader LLVM project as a guide.

- When testing the handling of command-line options, the `-###` option can be
  very useful. This prints the command-lines used to invoke `-cc1` or `-fc1`,
  but will not actually invoke those language-specific compilers (or the
  linker). However, command-line option diagnostics will be issued which can be
  used to check that the options were validated. This also improves testing
  times since it is much faster to print the command-lines than to invoke the
  underlying compiler.

- When testing Kitsune's language extensions, the high-level source code that
  is used should be kept to a bare minimum. The use of standard library headers
  and functions should be avoided if possible. For instance, to check a simple
  `for` loop, the empty loop below should be used if possible

  ```c
  void f(long n) {
    for (long i = 0; i < n; ++i) {}
  }
  ```

  The code below should be avoided since it requires `<stdio.h>`. This is
  because `stdio.h` may be in a non-standard location make it more difficult
  to craft a command line that will work on all systems.

  ```c
  #include <stdio.h>
  void f(long n) {
    for (long i = 0; i < n; ++i) {
      printf("Hello\n");
    }
  }
  ```

  If a non-empty loop body is required, prefer something closer to the code
  below that does not require "external" files.

  ```c
  void f(float *a, long n) {
    for (long i = 0; i < n; ++i) {
      a[i] = i;
    }
  }
  ```

- When testing LLVM-IR passes, do *not* use high-level source code as input.
  Instead, LLVM-IR should be used as input and the pass invoked directly using
  LLVM's [opt](https://llvm.org/docs/CommandGuide/opt.html) utility. To the
  extent possible, the LLVM-IR should be the bare minimum that triggers the
  exact behavior in the pass that is being tested. Tools such as
  [llvm-reduce](https://llvm.org/docs/CommandGuide/llvm-reduce.html) and
  Kitsune's [kit-enc](CommandGuide/kit-enc.md) may be useful in such cases.

- To test Kitsune's codegen passes, LLVM's
  [llc](https://llvm.org/docs/CommandGuide/llc.html) tool could be used. The
  `--tapir` option must be used to ensure that Kitsune's codegen passes are run.
  These passes can also be run explicitly using LLVM's
  [opt](https://llvm.org/docs/CommandGuide/opt.html) tool as well by explicitly
  adding them to the `-passes` command-line option.

Most core tests can be found in
`kitsune/test` and `kitsune/unittests`, but, historically, some may also be
found in `llvm/test` and `clang/test`. In general, new tests should be added to
`kitsune/test` or `kitsune/unittests`. However, if the change is not strictly
Kitsune-specific, it may be added to the `test/` directory of the relevant
subproject.

For example, the `--lto-emit-llvm` option was added to the MachO backend of
[lld](https://lld.llvm.org/). A corresponding option was present in both the
ELF and COFF backends, but was missing for MachO. This option was required
for Kitsune's LTO tests in `kitsune/test/lto`. However, the option is not
inherently Kitsune-specific and could, in principle, be useful to "vanilla"
`lld`. As a result, tests for this new option were added to `lld/test/MachO`,
_not_ `kitsune/test/tools/lld` [^1].

The tests in `kitsune/test` are designed to run with LLVM's
[lit](https://llvm.org/docs/CommandGuide/lit.html) tool and will be hereafter
referred to as "lit tests".

### Running the core tests

The core tests can be run from the build directory after
[building Kitsune](GettingStarted.md#build). The core tests in `kitsune/test`
and `kitsune-unittests` can be run using the `check-kitsune` target. The
example below shows how to run these using both the `Ninja` and `Unix Makefiles`
cmake generators.

```
ninja check-kitsune
make check-kitsune
```

If you wish to run just the unittests, use the `check-kitsune-unit` target.

```
ninja check-kitsune-unit
```

The tests in a single subdirectory of `kitsune/test` can be run if needed. For
example, the tests in `kitsune/test/tools/kit-mbc` can be run as follows.

```
ninja check-kitsune-tools-kit-mbc
```

In this case, the tests are in the `tools/kit-mbc` subdirectory of
`kitsune/test`. LLVM's build system generates a target obtained by replacing
the path separator `/` with `-`. This can be helpful during development when
one wishes to run the tests for a single tapir target. For instance, tests for
the [cuda](tapir-targets-cuda) tapir target can be run as follows since these
tests are in `kitsune/test/tapir/cuda`.

```
ninja check-kitsune-tapir-cuda
```

For convenience, Kitsune's build system provides some shorter aliases. The
tests for the [cuda](tapir-targets-cuda) can also be run as follows.

```
ninja check-kitsune-cuda
```

Individual lit tests can also be run by invoking `llvm-lit` directly. Here
`<file>` is a single-file test. These do not have to be in `kitsune/test`.
Any test in Kitsune's repository - including those in `llvm/test`, `clang/test`
and so on can be run this way.

```
llvm-lit <file>
```

When diagnosing test failures, the
[-v](https://llvm.org/docs/CommandGuide/lit.html#cmdoption-lit-v) option
provided by `llvm-lit` can be useful. Another option that is occasionally useful
is [-a](https://llvm.org/docs/CommandGuide/lit.html#cmdoption-lit-a) which can
be used to examine the `RUN` lines in a test

```
llvm-lit -av <file>
```

The lit tests in a directory by specifying the path to a directory in the
command above. This can be useful when diagnosing performance issues with
tests. For example, the
[`--time-tests`](https://llvm.org/docs/CommandGuide/lit.html#cmdoption-lit-time-tests)
option provided by `llvm-lit` can be used to obtain a report of the time taken
to run each test in a directory.

```
llvm-lit --time-tests <dir>
```

[^1]: In such cases, we do attempt to upstream such code if possible. In this particular case, we did submit a [PR](https://github.com/llvm/llvm-project/pull/170355) to LLVM.

[^2]: For example, if a test for pass A requires pass B to be run, an error in pass B will also result in the failure of the test of pass A. This can result in wasted time spent debugging pass A.

## End-to-end Tests

The "end-to-end" tests run the entire compiler pipeline and produce functioning
executables. These are used to check that the resulting executable both runs
and produces the correct answer [^3].

```{important}
The main Kitsune repository does *not* contain any end-to-end tests.
```


[^3]: For code that makes extensive use of floating point arithmetic, the results produced when compiling with Kitsune may not be bitwise-identical to those produced by other compilers. This has to do with the fundamental limitations of floating-point arithemtic and its interaction with compiler optimizations.
