The tests for Kitsune are scattered in a number of different places. This is 
mainly intended for tests for the frontend and lowering to LLVM-IR for both the
C/C++ and Fortran frontends. In general, Kitsune-specific tests should be added
here rather than `clang/test` or `flang/test`. However, middle and backend 
tests can be found both here and and in `llvm/test`. There is no firm rule on 
which tests should go here verses in `llvm/test`. In general, if the property
being tested can be done with high-level (C/C++/Fortran) source as input, we 
should prefer to do that here. The `RUN` line will likely be of the form:
```
RUN: %<frontend> -ftapir=<tapir-target> -O2 -S -emit-llvm %s | FileCheck ...
```
In some cases, this is not possible and it is more convenient to use LLVM-IR 
as input - for example to check something specific in a Kitsune-centric LLVM
pass. In that case, it is better to add the test to an appropriate subdirectory
of `llvm/test`. 

The tests in this directory are organized somewhat loosely into directories for
the frontends and those for specific tapir targets. At the time of writing, 
clang and flang share a driver, and therefore, so do kitcc/kit++ and kitfc. The
command-line option processing tests are generally in `driver/` though some can
be found in the directories for the specific frontends if the option is specific
to that frontend. 

When using high-level source as input for a test of a Tapir target, the test 
should be placed in the directory for that Tapir target. It may be necessary to
have both C/C++ and Fortran implementations of the test in such cases. If that 
is too onerous, one can write the test in LLVM-IR and add it to `llvm/test`if it
is reasonable to assume that the IR is representative of what both the `kitcc`
and `kitfc` frontends would produce.
