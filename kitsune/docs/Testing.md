# Testing Strategy

This document describes how Kitsune's tests are organized and how various
parts of Kitsune should be tested.

## Overview

Tests for Kitsune are split into two categories, [core](#core-tests) tests and
[end-to-end](#end-to-end-tests) tests. As a developer, both the core tests and
the end-to-end tests should be run regularly. It is not uncommon for the core
tests to pass but for the end-to-end tests to fail.

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
unrelated areas of the compiler may manifest as spurious test failures. For
example, if a test for pass A requires pass B to be run, an error in pass B will
also result in the failure of the test of pass A. This can result in wasted time
spent debugging pass A.

### Writing Core Tests

We follow the pattern established in LLVM to write core tests. These make
extensive use of LLVM's
[lit](https://llvm.org/docs/CommandGuide/lit.html) and
[FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) utilities. Since
`lit` is the tool that runs these tests, we will refer to them as "lit tests".

```{note}
The tests do not always have to use FileCheck, but the overwhelming majority of
tests will do so.
```

The majority of "lit tests" consist of a single file. The files will certainly
contain one or more lit directives - the most common being `RUN:`. Most files
will also contain either high-level source or LLVM assembly, but many will
only contain lit directives. The `RUN:` directives will typically consists of
command-lines to be executed.

In some cases, the files may require an additional files as input. These will
typically be present in an `input/` directory in the same directory as the
"lit test". Tests that are primarily focused on the linker such as those in
`kitsune/test/lto/` and `kitsune/test/tools/lld` may require multiple input
files.

This is a very high-level overview of Kitsune's (really, it is LLVM's) testing
infrastructure. A detailed description of these is outside the scope of this
document.
[LLVM's testing infrastructure guide](https://llvm.org/docs/TestingGuide.html)
contains more information. We recommend using the existing tests - both
Kitsune-specific and those in the broader LLVM project - as a guide.

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

Some tips for writing core tests are presented here. These are not intended to
be comprehensive.

[^1]: We do attempt to upstream such code. In this particular case, we did submit a [PR](https://github.com/llvm/llvm-project/pull/170355) to LLVM.

#### Testing the Driver

When testing the handling of command-line options, the `-###` option can be
very useful. This prints the command-lines used to invoke `-cc1` or `-fc1`,
but will not actually invoke those language-specific compilers (or the
linker). However, command-line option diagnostics will be issued which can be
used to check that the options were validated. This also improves testing
times since it is much faster to print the command-lines than to invoke the
underlying compiler.

#### Testing Language Extensions

When testing Kitsune's language extensions, the high-level source code that
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

#### Testing Frontends

Some of Kitsune's language extensions apply to both C and C++. Strictly
speaking, we should test that these are handled correctly by the C and C++
frontends, that is `kitcc` and `kit++` respectively. However, we generally
prefer to test with only the C++ frontend for the following reasons:

1. Most of the relevant frontend code operates on the language-independent
   [clang AST](https://clang.llvm.org/docs/IntroductionToTheClangAST.html).
   Therefore, it is often sufficient to test with just one frontend.

2. Both C and C++ frontends are _always_ enabled, so it is reasonable to pick
   one of the two and just test with that. We expect the majority of Kitsune's
   users to be interested in C++, not C, so we prefer to test with that
   frontend.

Some of Kitsune's builtins are supported in both C and C++. The lowering of
these to LLVM-IR is generally tested with the C frontend. Since name-mangling
does not occur in C code, writing `FileCheck` checks is often easier for C
code.

#### Testing LLVM-IR Passes

When testing LLVM-IR passes, do *not* use high-level source code as input.
Instead, LLVM-IR should be used as input and the pass invoked directly using
LLVM's [opt](https://llvm.org/docs/CommandGuide/opt.html) utility. To the
extent possible, the LLVM-IR should be the bare minimum that triggers the
exact behavior in the pass that is being tested. Tools such as
[llvm-reduce](https://llvm.org/docs/CommandGuide/llvm-reduce.html) and
Kitsune's [kit-enc](CommandGuide/kit-enc.md) may be useful in such cases.

Ideally, a pass should be tested on its own i.e. only the pass under test should
be run on input crafted to test some specific behavior of the pass. The example
below shows how to run only the loop-spawning pass.

```
opt -passes='loop-spawning' in.ll
```

In some cases, however, it may not be possible - or convenient - to run a
single pass. This is particularly true for the GPU-centric passes in
Kitsune's [pass pipeline](PassPipeline.md). These passes typically require a
specific sequence of passes to have run in order to work correctly. One could,
in principle, craft LLVM-IR that would allow these passes to be run on their
own, but it may not be worth the effort. In such cases, one will often instruct
`opt` to run one or more passes. A fairly common pattern in Kitsune's tests is
to run the tapir pipeline followed by a specific pass.

```
opt -passses='tapir-lowering<O1>,kit-cgfb' in.ll
```

If it is possible to craft IR that can be used to run just a single pass, that
should be preferred. Understanding the nuances of Kitsune's
[pass pipelines](PassPipeline.md) will help in deciding when deciding how to
test a specific pass.

#### Testing Embedded Bitcode Passes

Some tapir targets, particularly [cuda](tapir-targets-cuda) and
[hip](tapir-targets-hip) use [embedded bitcode](EmbeddedBitcode.md). Some
passes in Kitsune's pass pipeline operate exclusively on this embedded bitcode.
Testing these requires a slightly different approach since the lit tests
typically expect that the LLVM module that is passed to `opt` is what is
being tested.

To test an embedded bitcode pass, we need to craft LLVM-IR that is appropriate
for the pass being tested, then embed that into a host module that can be passed
to `opt`. The [kit-enc](CommandGuide/kit-enc.md) utility can be used for this. A
simple example of such a lit test is shown below

```llvm
; RUN: %kit-enc %s \
; RUN:     | opt -passes='my-mbc-pass' \
; RUN:     | kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: <check-strings>;

define void @f() {
  ret void
}
```

In the example above, the LLVM-IR module consisting of a single function named
`@f` is embedded into an empty host module in the call to `kit-enc`. This
module is then piped into `opt` which runs the embedded bitcode pass
`my-mbc-pass`. This pass will run on the embedded bitcode and update it as
needed. The output of `opt` will be a module containing the updated embedded
bitcode. This is passed to [kit-mbc](CommandGuide/kit-mbc.md) which will
extract the embedded bitcode from the host module and write it as human-readable
LLVM assembly to stdout. This, in turn, is piped into `FileCheck`.

#### Repeated Tests

When testing embedded bitcode passes, one is required to specify which tapir
target generated the embedded bitcode. Some of these passes behave exactly the
same regardless of the tapir target that generated the embedded bitcode. In
such cases, one could, in principle, pick an arbitrary tapir target that uses
embedded bitcode to test the pass. However, these tapir targets are not
"universal", so one could choose to not build the target. In this case, the
test would not run raising the likelihood of inadvertently introducing bugs if
one is developing Kitsune with only a subset of the supported tapir targets
enabled.

Therefore, we recommend testing embedded bitcode passes in _each_ of
the tapir targets that use embedded bitcode. This does result in very similar
tests that differ only in the tapir target that is used. An alternative would be
to use
[conditional substitution](https://llvm.org/docs/TestingGuide.html#substitutions).
However, these are very limited in functionality; writing complex `RUN:` directives is difficult and the result is hard to read and maintain.

A similar issue arises when testing the handling of tapir-target-specific
configuration files. While much of the code that deals with these is common to
all tapir targets, a small amount isn't. While this could be tested with, for
instance, a unit test, testing the functionality separately with each tapir
target is more convenient.

The Fortran driver, `kitfc`, shares a lot of code with the C and C++ frontends.
However, command-line options must be explicitly enabled for each driver.
Therefore, for tests of command-line options, a Fortran test must be written to
ensure that the option is enabled (or disabled) in Fortran.

### Running the Core Tests

The core tests can be run from the build directory after
[building Kitsune](GettingStarted.md#build). The core tests in `kitsune/test`
and `kitsune-unittests` can be run using the `check-kitsune` target. The
example below shows how to run these if the `Ninja` generator was used when
configuring Kitsune.

```
ninja check-kitsune
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
tests for the [cuda](tapir-targets-cuda) tapir target can also be run as shown
below. This pattern can also be used to run the tests for the other
[supported](tapir-targets-supported) tapir targets.

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

## End-to-end Tests

The "end-to-end" tests run the entire compiler pipeline and produce functioning
executables. These are used to check that the resulting executable both runs
and produces the correct answer [^2].

```{important}
The main Kitsune repository does *not* contain _any_ end-to-end tests.
```

Building and running end-to-end tests requires a complete development
toolchain, including standard libraries, linkers and, on some platforms,
a dynamic linker (also known as a loader). Even within the
limited number of platforms that Kitsune supports, it is easy to find systems
with non-standard sysroots that make writing reliable build scripts for such
tests very difficult. On most HPC clusters for instance, Kitsune cannot always
be run from the build directory because the C++ standard libraries found by the
default dynamic linker (loader) are often incompatible. In such cases, Kitsune
needs to be explicitly provided a functional alternative sysroot - including a
loader in a non-standard location. Adding configuration files to the build
directory is a possibility, but one risks breaking upstream tests, such as those
in `clang/test` and `flang/test`. In such cases,, installing Kitsune and
providing a configuration file to be used from the install directory is known
to work well.

```{important}
End-to-end tests must not be added to the core tests in `kitsune/test`.
```

All end-to-end tests are in the
[Kitsune Test Suite](https://github.com/tarunprabhu/kitsune-test-suite). See the
[documentation](KitsuneTestSuite.md) for the test suite for details on building
and running the tests.

```{note}
There are currently no guidelines on what constitutes a "good" end-to-end test.
When possible, if an application uncovers a bug in Kitsune, a
[core test](#core-tests) that exercises the buggy code should be crafted and
added to Kitsune's repository. An end-to-end test should only be added if it
exercises a complex path through Kitsune that cannot be reasonably tested in a
core test.
```

[^2]: For code that makes extensive use of floating point arithmetic, the results produced when compiling with Kitsune may not be bitwise-identical to those produced by other compilers. This has to do with the fundamental limitations of floating-point arithemtic and its interaction with compiler optimizations.

## Runtime tests

Currently, Kitsune's runtime has very limited testing. It largely relies on the
end-to-end tests in the [Kitsune test suite](KitsuneTestSuite.md) to test for
correctness, but these are, naturally, extremely coarse-grained.

```{note}
When fine-grained tests are added to Kitsune's runtime, they will be described
here.
```
