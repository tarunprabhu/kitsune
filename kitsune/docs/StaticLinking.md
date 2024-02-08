# Static Linking

Currently, there are several limitations that prevent Kitsune from producing
fully static executables in all cases. This document describes Kitsune's
support for static linking and its limitations.

## Static Linking of Kitsune's Runtime

When building Kitsune, a static archive of Kitsune's runtime, kitrt, is
automatically built. This is always named `libitrt.a`. By default, however, the
dynamic shared object (DSO) is automatically linked when compiling with
Kitsune. This is named `libkitrt.dylib` on MacOSX and `libkitrt.so` on other
supported platforms[^1]. Consider the `a.out` executable compiled with Kitsune
as shown
below.

```
kitcc -o a.out -O1 --tapir=opencilk in.c
```

On Linux, running the `ldd` command on `a.out` may produce something similar to
the following

```
ldd ./a.out
        linux-vdso.so.1
        libkitrt.so => /path/to/lib/clang/<NN>/lib/libkitrt.sp
        libamdhip64.so => /opt/rocm/lib/libamdhip64.so
        libcuda.so => /usr/lib/libcuda.so
        libc.so => /usr/lib/libc.so
        ...
```

Here, we can see that `libkitrt.so` was specified at link-time. At run-time,
this library, and others must be in a path known to the dynamic linker (also
known as a loader). The `...` in the output above indicates that some output has
been omitted for clarity.

In order to link against the static archive, `libkitrt.a`, the
`-static-libkitrt` option should be used.

```
kitcc -o a.out -O1 --tapir=opencilk in.c -static-libkitrt
```

In this case, _only_ `libkitrt.a` will be linked statically. If one runs `ldd`
on `a.out`, there will be no reference to `libkitrt`.

```
ldd ./a.out
        linux-vdso.so.1
        libamdhip64.so => /opt/rocm/lib/libamdhip64.so
        libcuda.so => /usr/lib/libcuda.so
        libc.so => /usr/lib/libc.so
        ...
```

## Full Static Linking

In the listings above. note that even though the tapir target was set to
[opencilk](tapir-targets-opencilk), both `libamdhip64.so` and `libcuda.so` were
linked into the final executable. These are required by the
[cuda](tapir-targets-cuda) and [hip](tapir-targets-hip) tapir targets.
The `opencilk` tapir target only requires
[cheetah](https://github.com/opencilk/Cheetah), the runtime system of
[OpenCilk](https://www.opencilk.org).

This is a known limitation of Kitsune's runtime, `kitrt`.
Kitsune's runtime requires the dependencies of _all_ tapir targets that were
built with Kitsune to be available whenever linking against `libkitrt`. In this
case, even though we know that the `opencilk` tapir target will not require
anything from `libcuda.so` or `libamdhip64.so`, the two libraries must be
available both at link-time and at run-time because `libkitrt` is always linked
in when a tapir target is specified at compile-time.

```{note}
We do intend to remove this limitation at some point, but there are no immediate
plans to do so.
```

This impacts Kitsune's ability to produce a fully statically linked executable.
Normally, in order to create a statically linked executable, the `-static`
option is used. However, with Kitsune, this is likely to fail with the following
error

```
kitcc -o /tmp/a.out -O1 --tapir=opencilk in.c -static

ld: cannot find -lamdhip64: No such file or directory
ld: have you installed the static version of the amdhip64 library ?
ld: cannot find -lcuda: No such file or directory
ld: have you installed the static version of the cuda library ?
```

Because the NVIDIA and AMD drivers must be present when using Kitsune's runtime,
the linker searches for `libcuda.a` and `libamdhip64.a`. On the
system on which this command was run, these were not available and the static
link, therefore, failed.

```{note}
We are not aware of any system on which static archives of the driver are
available. We have not been able to find an authoritative source that says that
such archives are not distributed by NVIDIA and AMD respectively. However, we
have not found these to be available on any system that we have used.
```

### Experimental Workaround for Full Static Linking

In order to build a fully statically executable for use with, say, the
`opencilk` tapir target, the `cuda` and `hip` tapir targets must _not_ be
enabled when building Kitsune. This can be achieved by omitting these tapir
targets from the `-DKITSUNE_ENABLE_TAPIR_TARGETS` configure-time option when
building Kitsune. For this to work, `-DKITSUNE_ENABLE_TAPIR_TARGETS` _must_ be
provided. If this configuration option is omitted, both the `cuda` and `hip`
tapir targets will be built. For more information, see
[this section](GettingStarted.md#configure).

Once Kitsune has been built without either the `cuda` or `hip` tapir targets,
the `-static` command-line option should be sufficient to produce a fully
statically linked executable.

This has been tested with the [opencilk](tapir-targets-opencilk),
[pthreads](tapir-targets-pthreads) and [serial](tapir-targets-serial) tapir
targets.

[^1]: Windows is not currently supported, so `libkitrt.dll` is never built
