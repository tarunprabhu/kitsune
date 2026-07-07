# Lit tests for Kitsune's runtime

The tests here are essentially "system" tests that exercise several parts of
Kitsune's runtime, and are, therefore, not suitable as unit tests. In most
cases, they rely on the output of the runtime's verbose mode (enabled by setting
the `KIT_VERBOSE` environment variable).

These are `lit` tests, but are unlike the `lit` tests in the rest of the repo.
While the test files contain lit directives, they are not usually inputs to an
LLVM tool. Instead, each source file is compiled to an executable that is
invoked from at least one `RUN` directive in the file. The special `lit`
substitution `%exe` is the path to this executable. The example below shows what
a typical test in this directory might look like:

```c
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s

// Include kitrt headers, if available.

int main(int argc, char *argv[]) {
  // Use Kitsune's runtime here.
}
```

Note that, in some cases, we will have to provide declarations for Kitsune's
runtime functions since they may not be declared in any header files. This is
intentional since `libkitrt` is only intended for use by the Kitsune compiler,
which has no use for the headers. In the future, if we do provide header files,
those can be included instead of declaring the functions explicitly.
