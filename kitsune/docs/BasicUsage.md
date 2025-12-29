# Basic Usage

Here, we provide some very simple examples of using Kitsune's drivers and
language extensions.

```{warning}
Kitsune is under active development. The language extensions in particular
should be considered experimental and subject to change. The goal is to
eventually have definitive examples in the
{{'[`kitsune/examples/`](https://{}/{}/kitsune/tree/{}/kitsune/examples)'.format(kitsune_repo_host, kitsune_repo_owner, kitsune_repo_branch)}}
directory of Kitsune's source repository.
Currently, the only examples are in the
[Kitsune Test Suite](https://github.com/tarunprabhu/kitsune-test-suite/tree/main/Kitsune)
and in the
{{'[`kitsune/experiments/`](https://{}/{}/kitsune/tree/{}/kitsune/experiments)'.format(kitsune_repo_host, kitsune_repo_owner, kitsune_repo_branch)}}.
directory in Kitsune's source repository.
However, the code in both locations is not exactly "beginner-friendly".
```

## Introduction

The core of Kitsune's capabilities involve optimizing explicitly parallel loops.
Kitsune provides an explicitly parallel [`forall`](extensions-cxx-forall)
loop that can be used in C and C++. The
[semantics](extensions-cxx-forall-semantics) and
[limitations](extensions-cxx-forall-limitations) are discussed elsewhere.

```kitxx
#include <kitsune.h>

int main(int argc, char* argv[]) {
    long n = std::strtol(argv[1]);
    long c = std::strtol(argv[2]);
    long *a = nullptr;

    a = new long[n];
    forall (long i = 0; i < n; ++i) {
        a[i] = c;
    }
    delete[] a;

    return 0;
}
```

The code above is a simple example of a program that initializes an array using
a parallel loop.

```{important}
Although the `forall` loop is a keyword in the language, note that we have
included `kitsune.h` here. This is required, otherwise, `forall` will not be
recognized as a keyword.

This must be the last header included in a source file. If any headers are
included after `kitsune.h`, there is a risk that the code will not compile.
```

## Compiling for the CPU

Kitsune currently supports compiling C, C++ and Fortran. The appropriate driver
must be used in each case. These are named `kitcc`, `kit++`, and `kitfc` for
C, C++ and Fortran respectively. Unlike `clang` which allows compiling both C
and C++ with the same driver, compiling C++ code with `kitcc` and vice versa
may result in an error.

A [tapir target](glossary-tapir-target) must be explicitly set in order to
recognize Kitsune's language constructs. One could compile the code above with,
for instance, the [pthreads](tapir-targets-pthreads) tapir target, as shown
below

```shell
kit++ --tapir=pthreads -O1 ...
```

```{important}
Currently, Kitsune requires optimizations in order to work. Therefore, if the
`--tapir` option is used, at least `-O1` must be specified.
```


## Compiling for the GPU

This code, as written, will not work with the GPU-centric tapir targets. This is
because the arrays used in the body of the `forall` loop are allocated in
[host](glossary-host) memory. This memory cannot be accessed directly from a
GPU. All objects - including arrays - accessed in a `forall` loop that is
compiled for execution on a GPU must be allocated in unified memory [^1][^2].

Kitsune provides custom [memory allocation functions](MemoryManagement.md) that
can be used for this. The code below shows how the example above could be
modified to use these.

```kitxx
#include <kitsune.h>

int main(int argc, char* argv[]) {
    long n = std::strtol(argv[1]);
    long c = std::strtol(argv[2]);
    kitsune::mobile_ptr<long> a;

    a.alloc(n);
    forall (long i = 0; i < n; ++i) {
        a[i] = c;
    }
    a.free();

    return 0;
}
```

The use of Kitsune's `mobile_ptr` class will work when compiling with any
tapir target. When using a CPU-centric tapir target such as
[pthreads](tapir-targets-pthreads), or [opencilk](tapir-targets-opencilk),
memory will be allocated using host memory as one would normally expect using
`malloc` from the standard C library.

```{tip}
For maximum flexibility, when using Kitsune, we recommend always using
Kitsune's dedicated memory management functions. That way, one can compile for
both CPU and GPU without needing to modify the code.
```

### Compiling for NVIDIA GPU's

The [cuda](tapir-targets-cuda) tapir target must be used to run the `forall`
loop on an NVIDIA GPU.

```shell
kit++ --tapir=cuda -O1 ...
```

Currently, Kitsune can only compile code for a specific NVIDIA GPU. If a GPU
architecture has not been explicitly specified, Kitsune will compile for the
NVIDIA GPU that is present on the system. If an NVIDIA GPU was not detected,
a default architecture will be used. The current default is
{{'`{}`'.format(kitsune_cuda_arch_default)}}.

An explicit architecture can be specified as follows

```shell
kit++ --tapir=cuda --tapir-cuda-arch=<architecture> -O1 ...
```

See the [C/C++ command line reference](KitClangOptionsDoc) for more options
that can be used with the `cuda` tapir target. These options will start with
`tapir-cuda-`. Other options starting with `tapir` can also be used, but the
ones starting with `tapir-cuda` are specific to the `cuda` tapir target.

### Compiling for AMD GPU's

The [hip](tapir-targets-hip) tapir target must be used to run the `forall`
loop on an AMD GPU.

```shell
kit++ --tapir=hip -O1 ...
```

Currently, Kitsune can only compile code for a specific AMD GPU. If a GPU
architecture has not been explicitly specified, Kitsune will compile for the
AMD GPU that is present on the system. If an AMD GPU was not detected,
a default architecture will be used. The current default is
{{'`{}`'.format(kitsune_hip_arch_default)}}.

An explicit architecture can be specified as follows

```shell
kit++ --tapir=hip --tapir-hip-arch=<architecture> -O1 ...
```

See the [C/C++ command line reference](KitClangOptionsDoc) for more options
that can be used with the `hip` tapir target. These options will start with
`tapir-hip-`. Other options starting with `tapir` can also be used, but the
ones starting with `tapir-hip-` are specific to the `hip` tapir target.


[^1]: [https://developer.nvidia.com/blog/unified-memory-cuda-beginners/](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
[^2]: [https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html)
