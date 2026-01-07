# Using LLVM tools with Kitsune

Some LLVM tools have been extended with Kitsune-specific functionality. These
are described here.

First, we show how the compilation process can be decomposed into a number of
steps, where each step is executed manually using LLVM's tools. We do not
describe each step below in detail, in particular the use of `opt` and `llc`.
Please see the corresponding sections for [opt](llvm-tools-opt) and
[llc](llvm-tools-llc) in this document for details.

The listing below is a simple example where an array is initialized with a
constant using a parallel loop.

```kitc
#include <stdio.h>
#include <stdlib.h>
#include <kitsune.h>

int main(int argc, char* argv[]) {
  int n = argc > 1 ? atoi(argv[1]) : 256;

  int* a = (int*)malloc(sizeof(int) * n);
  forall (int i = 0; i < n; ++i)
    a[i] = i;

  for (int i = 0; i < n; ++i)
    printf("%d ", a[i]);
  printf("\n");

  return 0;
}
```

We first generate LLVM-IR from this code. For this example, we use the
[pthreads](tapir-targets-pthreads) tapir target since this is the most basic
tapir target that can exploit parallelism.

```bash
clang --tapir=pthreads -S -emit-llvm -o in.ll -O1 -mllvm -disable-llvm-optzns in.c
```

At the time of writing, if a tapir target is specified, Kitsune requires
optimizations to be enabled. Therefore, we provide `-O1`. However, we do
not want the optimizations to actually run, so we also add
`-mllvm -disable-llvm-optzns` to disable optimizations directly in the
[middle-end](glossary-middle-end) of the compiler.

`in.ll` should contain code similar to the following, indicating that it
contains [tapir loops](glossary-tapir-loop).

```kitll
detach within %syncreg, label %forall.body, label %forall.inc
```

We must now run [Kitsune's pass pipeline](PassPipeline.md) on this IR. We
write this optimized output to `in.opt.ll`. Note that we have used `-S` here
to generate human-readable [LLVM assembly](glossary-llvm-assembly). This is
for convenience since we may wish to examine the contains of `in.opt.ll`.

```bash
opt --tapir=pthreads -O3 -S -o in.opt.ll in.ll
```

Now that we have optimized LLVM-IR, we can compile this to machine code.

```bash
llc --tapir=pthreads -O3 -filetype=obj -o in.o in.opt.ll
```

This object file can now be linked to produce an executable. While we could
have used a linker directly, it is much more convenient to use to the driver
and let it invoke the linker internally. Among other things, this will ensure
that Kitsune's runtime is linked in correctly. Here, `--no-pie` must be passed
to the linker since the code was not compiled with `-fPIE`.

```bash
kitcc --tapir=pthreads -O3 -o a.out in.o -Xlinker --no-pie
```


(llvm-tools-llc)=
## llc

[llc](https://llvm.org/docs/CommandGuide/llc.html) is a tool that compiles
[LLVM-IR](glossary-llvm-ir) to lower-level code. It is capable of generating
both human-readable _machine_ assembly [^1] (not to be confused with
[LLVM assembly](glossary-llvm-assembly)) as well as binary object files [^2].
Like [opt](llvm-tools-opt), `llc` runs a series of passes on the input.
Unlike in `opt`, where the majority of passes that are run are
target-independent, most passes run by `llc` are target-specific. They are part
of LLVM's [codegen](glossary-codegen) pipeline.

When compiling code containing [tapir loops](glossary-tapir-loop) and other
constructs containing [tapir instructions](glossary-tapir-instruction), a
number of Kitsune-specific codegen passes must be run. These are only run
when a specific tapir target is specified. As with Kitsune's drivers, this is
done by passing the `--tapir` command-line option. In addition to `--tapir`,
all command-line options beginning with `--tapir-` that are supported by the
[drivers](glossary-driver) are also supported by `llc`. A complete list of these
options can be found [here](KitClangOptionsDoc). The example below shows the use
of the [pthreads](tapir-targets-pthreads) tapir target.

```bash
llc --tapir=pthreads ...
```

The input file provided to `llc` must contain either
[LLVM bitcode](glossary-bitcode) or [LLVM assembly](glossary-llvm-assembly).

```{important}
When the `--tapir` option is provided, _only_ the codegen pass pipeline is
run. Passes such as loop-spawning are *not* run. Therefore, depending on the
tapir target that was specified, any tapir instructions in the IR will not be
replaced. These will eventually make it to the latter stages of the codegen
pipeline where they will likely result in a crash.
```

Depending on the tapir target being used, a number of other
options will also be required. For example, the GPU-centric tapir targets
require the architecture of the GPU for which to generate code while the
[opencilk](tapir-targets-opencilk) requires the path to the
[OpenCilk](https://www.opencilk.org) runtime bitcode file. When using Kitsune's
drivers such as `kitcc`, `kit++` and `kitfc`, appropriate values for these
options will be calculated automatically. However, this is not the case with
`llc`. The table below lists the required options for various tapir targets.

```{table}
| Tapir<br>Target | Required<br>Options |
| :-------------- | :-----------------: |
| cuda | `--tapir-cuda-arch`<br>`--tapir-cuda-runtime-bc` |
| custom | `--tapir-plugin` |
| hip | `--tapir-hip-arch`<br>`--tapir-hip-runtime-bcs` |
| nolo | |
| opencilk | `--tapir-opencilk-runtime-bc` |
| pthreads | |
| serial | |
```

One possible use of `llc` is to compile the output of optimized IR produced by
`opt` to object code. The example below shows how this might be done with the
`cuda` tapir target. Here, the `in.ll` file will most likely have been generated
by Kitsune's drivers from high-level source containing parallel loops (or
other constructs that are lowered to LLVM-IR containing tapir instructions).

```bash
opt --tapir=cuda \
    --tapir-cuda-arch=sm_86 \
    --tapir-cuda-runtime-bc=/path/to/cuda/libdevice.bc \
    -O3 -S -o in.opt.ll \
    in.ll

llc --tapir=cuda \
    --tapir-cuda-arch=sm_86 \
    --tapir-cuda-runtime-bc=/path/to/cuda/libdevice.bc \
    -O3 -filetype=obj -o in.o \
    in.opt.ll
```

In the invocation of `opt` in the listing above, specifying a tapir target
together with an optimization level that is greater than 0 ensures that the
Kitsune passes that lower tapir loops are run. The output of the command,
`in.opt.ll` will contain embedded bitcode for the GPU architecture that was
specified, in this case, `sm_86`. This file is then compiled to an object file
by `llc`. Note that rather than use temporary file, the output of `opt` could
have been passed to `llc` directly as shown below.

```bash
cat in.ll \
| opt --tapir=cuda \
      --tapir-cuda-arch=sm_86 \
      --tapir-cuda-runtime-bc=/path/to/cuda/libdevice.bc \
      -O3 \
| llc --tapir=cuda \
      --tapir-cuda-arch=sm_86 \
      --tapir-cuda-runtime-bc=/path/to/cuda/libdevice.bc \
      -O3 -filetype=obj -o in.o
```

[^1]: The human-readable assembly can be passed to an assembler to generate object code. Like the binary object directly generated by llc, this can be passed to a linker to generate an executable or a dynamic shared object.

[^2]: The binary object code can be passed to a linker to generate an executable or a dynamic shared object.


(llvm-tools-opt)=
## opt

[opt](https://llvm.org/docs/CommandGuide/opt.html) is a tool that can be used
to run one or more passes on [LLVM-IR](glossary-llvm-ir). These passes are
typically target-independent. There are a number of ways that they can be run.
Individual passes, or sequences of passes, can be run by adding them to the
`-passes` command-line option. For instance, the example below will run the
[sroa](https://llvm.org/docs/Passes.html#sroa-scalar-replacement-of-aggregates),
[tapir-indvars](passes-tapir-indvars) and
[loop-stripmine](passes-loop-stripmine) passes.


```bash
opt -passes='sroa,tapir-indvars,loop-stripmine' ...
```

[Meta-passes](glossary-meta-pass) can also be specified here which will result
in a sequence of passes being run. For instance, the command below will run
Kitsune's [core pass pipeline](passes-kit-lowering) with an optimization level
of `-O2`.

```bash
opt -passes='kit-lowering<O2>' ...
```

Like the [drivers](glossary-driver), `opt` also accepts the standard
optimization levels, `-O0`, `-O1`, `-O2`, and `-O3`.

```bash
opt -O3 ...
```

In this case, the standard sequence of optimization passes will be run. The
Kitsune-specific passes will _not_ be run unless a
[tapir target](glossary-tapir-target) is specified. As with the drivers, this is
done by passing the `--tapir` command-line option. In addition to `--tapir`,
all command-line options beginning with `--tapir-` that are supported by the
drivers are also supported by `opt`. A complete list of these options can be
found [here](KitClangOptionsDoc). The example below shows the use
of the [pthreads](tapir-targets-pthreads) tapir target.

```bash
opt --tapir=pthreads ...
```

Depending on the tapir target being used, a number of other
options will also be required. For example, the GPU-centric tapir targets
require the architecture of the GPU for which to generate code while the
[opencilk](tapir-targets-opencilk) requires the path to the
[OpenCilk](https://www.opencilk.org) runtime bitcode file. When using Kitsune's
drivers such as `kitcc`, `kit++` and `kitfc`, appropriate values for these
options will be calculated automatically. However, this is not the case with
`llc`. The table below lists the required options for various tapir targets.

```{table}
| Tapir<br>Target | Required<br>Options |
| :-------------- | :-----------------: |
| cuda | `--tapir-cuda-arch`<br>`--tapir-cuda-runtime-bc` |
| custom | `--tapir-plugin` |
| hip | `--tapir-hip-arch`<br>`--tapir-hip-runtime-bcs` |
| nolo | |
| opencilk | `--tapir-opencilk-runtime-bc` |
| pthreads | |
| serial | |
```

The example below shows how `opt` might be used with the `cuda` tapir target.
Here, the `in.ll` file will most likely have been generated
by Kitsune's drivers from high-level source containing parallel loops (or
other constructs that are lowered to LLVM-IR containing tapir instructions).
Here, the output file, `in.opt.ll` will contain embedded bitcode to be compiled
to machine code for an NVIDIA GPU whose architecture is `sm_86`. The
[host](glossary-host) code will contain calls to Kitsune's runtime.
[llc](llvm-tools-llc) may be used to compile `in.opt.ll` to an object file.

```bash
cat in.ll \
| opt --tapir=cuda \
      --tapir-cuda-arch=sm_86 \
      --tapir-cuda-runtime-bc=/path/to/cuda/libdevice.bc \
      -O3 -S -o in.opt.ll
```
