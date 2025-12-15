# kit-config - Print Kitsune configuration information

## Synopsis

**kit-config** _option_ [_option..._]

## Description

**kit-config** is analogous to
[llvm-config](https://llvm.org/docs/CommandGuide/llvm-config.html) and provides
some information about
the specific Kitsune build. For example, this can used to query the tapir
targets and frontends that have been built as well as the locations of any
dependencies of the tapir targets (for instance, this can be used to determine
which [nvidia-cuda-toolkit](https://developer.nvidia.com/cuda/toolkit)
installation is used by the [cuda](tapir-targets-cuda) tapir target).

**kit-config** is _not_ a superset of `llvm-config` and cannot
be used to obtain the installation prefix, the build mode and other
configuration information that is available in `llvm-config`.

## Options

**--c**

: Has the C frontend been built. Prints ON or OFF.

**--c-frontend**

: Path to the C frontend.

**--cuda-prefix**

: Path to the [nvidia-cuda-toolkit](https://developer.nvidia.com/cuda/toolkit)
  used by the [cuda](tapir-targets-cuda) tapir target.

**--cxx**

: Has the C++ frontend been built. Prints ON or OFF.

**--cxx-frontend**

: Path to the C++ frontend.

**--fortran**

: Has the Fortran frontend been built. Prints ON or OFF.

**--fortran-frontend**

: Path to the Fortran frontend.

**--help**

: Print a summary of **kit-config** command-line options.

**--hip-prefix**

: Path to the [ROCm](https://rocm.docs.amd.com/en/latest/) installation used by
  the [hip](tapir-targets-hip) tapir target.

**--known-langs**

: Print all languages for which Kitsune frontends are available. Not all
  of these may have been enabled in this configuration.

**--known-tapir-targets**

: All known tapir targets. Not all of these may have been enabled in this
  configuration

**--langs**

: The languages for which Kitsune frontends have been enabled

**--tapir-targets**

: The tapir targets that have been built

**--version**

: Print the version of this program as well as the LLVM version on which
  Kitsune is based.

## Exit Status

If an error occurs, `kit-config` exits with a non-zero value. Otherwise, exit
with 0 to indicate success.

## Examples

One or more options can be provided. In this case, the options will be
processed in order and the result of each option will be on a separate line.

```
kit-config --langs --tapir-targets
C C++
cuda custom pthreads serial
```
