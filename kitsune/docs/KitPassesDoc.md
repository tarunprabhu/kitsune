# LLVM Pass Reference

This provides a high-level summary of the [analysis](glossary-analysis-pass),
[transformation](glossary-transformation-pass) and
[meta-passes](glossary-meta-pass)
that are part of Kitsune. The majority of passes that are run are the
standard LLVM passes, which are
[documented elsewhere](https://llvm.org/docs/Passes.html). We also include
passes that are part of [Tapir](faq-difference-kitsune-tapir) since they are
a critical part of Kitsune's optimization pipeline.

<!----------------------------------------------------------------------------->

(passes-kitsune)=
## Kitsune Passes

This describes those passes that have been developed specifically for Kitsune.
Kitsune uses the standard LLVM passes as well as passes from
[Tapir](passes-tapir).


(passes-kitsune-analysis)=
### Analysis Passes

This section describes Kitsune's analysis passes.


(passes-tapir-target-analysis)=
#### tapir-target-analysis

This pass owns the instances of the `TapirTarget` objects that are used by the
[loop-spawning](passes-loop-spawning) pass that transforms a tapir loop to use
a specific parallel runtime or architecture. It can also be used to query the
set of tapir targets that are required by a function or module.


(passes-kitsune-embeddeded-bitcode)=
### Embedded Bitcode Passes

This section describes Kitsune's
[embedded bitcode passes](glossary-embedded-bitcode-pass). While these are
technically transformation passes, we keep them separate since, from the
perspective of the [host module](glossary-host-module), they will only affect
the initializers of a few global variables. However, sine those initializers
contain [embedded bitcode](glossary-embedded-bitcode), the effect of these
passes can be very significant. See [here](EmbeddedBitcode.md) for a more
detailed discussion of embedded bitcode and how it is used in Kitsune.

(passes-emb-link-libdevice-bitcode)=
#### emb-link-libdevice-bitcode

Link suitable [device bitcode](glossary-libdevice-bitcode-file) modules into the
embedded bitcode.


(passes-emb-optimize)=
#### emb-optimize

Run the standard sequence of optimization passes on the embedded bitcode
modules. The specific sequence of passes to be run is determined by the
optimization level provided when compiling the code, or, by the parameter
passed to the `kit-lowering` meta-pass.


(passes-emb-prepare)=
#### emb-prepare

Prepare the embedded bitcode for code generation. This will carry out any
architecture-specific transformations, that are unrelated to optimizations,
and that have not already been carried out by the tapir targets that created the
embedded bitcode module. For instance, for AMDGPU kernels, the arguments of the
[kernel function](glossary-kernel-function) must be placed in a specific address
space, as must any alloca's in the kernel functions. In some cases, the calling
conventions of such functions must also be changed. This is usually run
relatively late in Kitsune's optimization pipeline.


(passes-resolve-libdevice-calls)=
#### emb-resolve-libdevice-calls

This will look for calls to functions that have vendor-provided device-specific
implementations and replace those calls with calls to the corresponding
device-specific implementations. These functions are typically, but not always,
from the C standard library. In some cases, bitcode files with these
device-specific implementations are provided by the vendors. Such bitcode files
are linked into the embedded bitcode module by the
[emb-link-libdevice-bitcode](passes-emb-link-libdevice-bitcode) pass. Even if
they are loaded by this pass, they are not linked.


(passes-kitsune-meta)=
### Meta-Passes

This section describes Kitsune's meta-passes i.e. a pass that executes a
specific sequence of passes.

(passes-kit-lowering)=
#### kit-lowering

This meta-pass can be used to run Kitsune's core
[pass pipeline](PassPipeline.md). This is primarily intended to be used with
[opt](https://llvm.org/CommandGuide/opt.html). This is a parameterized pass
that requires an optimization level to be provided. The parameter must be one
of `O0`, `O1`, `O2`, `O3`, `Os`, or `Oz`. The example below shows
how this pass can be run with `opt`.

```shell
opt -passes='kit-lowering<O2>' ...
```

(passes-kitsune-transformation)=
### Transformation Passes

This section describes Kitsune's transformation passes. If any embedded bitcode
is present in the module, these passes will not modify it. They may modify any
other part of the IR. The only transformations that these are permitted to carry
out on any embedded bitcode is to delete it entirely.


(passes-kit-annotate-tapir-loops)=
#### kit-annotate-tapir-loops

Adds attributes to tapir loops that are meant to be read by passes that run
later in the lowering pipeline. The specific annotations depend on the tapir
targets that are to be used.

Currently, when using the GPU-centric tapir targets, the pass identifies
tapir loop nests (where the outermost loop is a tapir loop) and annotates it
with the perfect tapir loop nest depth. All perfectly nested tapir loops within
each nest are annotated with their nesting level.


(passes-kit-cgfb)=
#### kit-cgfb

Compiles any embedded bitcode generated by the GPU-centric tapir targets to
appropriate machine code. Updates the initializer of the global variable that
is expected to contain this [device code](glossary-device-code). After this,
all embedded device code is removed.

(passes-kit-ctors)=
#### kit-ctors

Generates global constructors and destructors needed by Kitsune. These global
constructors initialize Kitsune's own runtime as well as any others such as
such as [ROCm](https://www.amd.com/en/products/software/rocm.html)
required by the [hip](tapir-targets-hip), set any environment variables that
may be required, and so on. The global destructors shutdown the runtime,
return any resources that they may hold back to the operating system, and so on.
While not all tapir targets require Kitsune's - or any other - runtime, this
pass will nevertheless always be run.

In addition to creating the constructors and destructors, this pass will
also create any any global variables needed by the global ctor. In the case
of the GPU tapir targets and associated runtimes, these include globals for
the fat binary, and the bundle that wraps the fat binary.

This pass should only be run once per module and should be run as late as
possible.


(passes-kit-kernel-properties)=
#### kit-kernel-properties

Computes metadata about GPU kernels that are launched by the program. This
metadata includes information about the instruction mix within the kernel - the
numbers of floating point, integer and memory operations in the kernel's IR.
In the future, this could be expanded to include anything else that could be
useful. The computed metadata is saved in the initializer of a global variable.
This pass will only have an effect if at least one GPU-centric tapir target has
been enabled.

This pass is run as late as possible in the pipeline to ensure that all
optimizations have already been run on the embedded bitcode.


(passes-kit-prefetch)=
#### kit-prefetch

This pass generates calls to initiate movement of data between
[host](glossary-host) and [device](glossary-device) - typically a GPU.
This will only generate calls to Kitsune's
[prefetch](llvm-intrinsics-async-prefetch-htod)
[intrinsics](llvm-intrinsics-async-prefetch-dtoh).
This is typically run early in Kitsune's post-tapir pipeline.


(passes-kit-lower-intrinsics)=
#### kit-lower-intrinsics

Some Kitsune-specific intrinsics can be replaced with a call to a function in
Kitsune's runtime. This pass performs that replacement.


(passes-kit-serialize-tapir-loops)=
#### kit-serialize-tapir-loops

Serializes tapir loops that cannot be profitably lowered by the
[loop-spawning](passes-loop-spawning) pass. Depending on the tapir targets being
used, no loops may be serialized.

For instance, when using the GPU tapir target, any loops that are below a
certain depth within a loop nest are serialized since they cannot be launched
from within the kernel, nor can they be lowered to a multi-dimensional GPU
kernel.


(passes-kit-strip-addr-spaces)=
#### kit-strip-addr-spaces

Replace pointers in Kitsune-specific address spaces with pointers in the
default address space. The same transformation is also performed on any
embedded bitcode modules. This is done because several
[back-ends](glossary-back-end) cannot handle Kitsune-specific address spaces.
This pass is typically added to the codegen pass pipeline, by which time, there
is no need for the Kitsune-specific address spaces.


<!----------------------------------------------------------------------------->

(passes-tapir)=
## Tapir Passes

This section describes passes that are part of the Tapir extensions to LLVM.


(passes-tapir-analysis)=
### Analysis Passes

This section describes Tapir's analysis passes.


(passes-tasks)=
#### tasks

This pass computes and provides information about the
[tapir tasks](glossary-tapir-task) in a function.


(passes-tapir-meta)=
### Meta-Passes

This section describes Tapir's meta-passes i.e. a pass that executes a
specific sequence of passes.


(passes-tapir-lowering)=
#### tapir-lowering

This meta-pass can be used to run Tapir's core pass pipeline. This runs all
the passes in the [tapir-lowering-loops](passes-tapir-lowering-loops) meta-pass
followed by a sequence of "cleanup" passes.
This is primarily intended to be used with `opt`. This is a parameterized pass
that requires an optimization level to be provided. The parameter must be one
of `O0`, `O1`, `O2`, `O3`, `Os`, or `Oz`. The example below shows
how this pass can be run with `opt`.

```shell
opt -passes='tapir-lowering<O2>' ...
```

(passes-tapir-lowering-loops)=
#### tapir-lowering-loops

This meta-pass can be used to run Tapir's core loop lowering passes. The
[tapir-lowering](passes-tapir-lowering) meta-pass runs all the passes in this
pipeline, followed by a sequence of "cleanup" passes. In most cases, it is
advisable to run the `tapir-lowering` meta-pass.
This is primarily intended to be used with `opt`. This is a parameterized pass
that requires an optimization level to be provided. The parameter must be one
of `O0`, `O1`, `O2`, `O3`, `Os`, or `Oz`. The example below shows
how this pass can be run with `opt`.

```shell
opt -passes='tapir-lowering-loops<O2>' ...
```


(passes-tapir-transformation)=
### Transformation Passes

This section describes Tapir's transformation passes.

(passes-loop-spawning)=
#### loop-spawning

This pass processes the [tapir loops](glossary-tapir-loop) in a function with
the appropriate tapir targets. Some tapir targets, such as
[opencilk](tapir-targets-opencilk), will do all of the work of transforming the
loop to use the appropriate runtime system - including inserting all the
required calls to the runtime functions. Others, like
[pthreads](tapir-targets-pthreads), will transform the loops and insert calls to
Kitsune's intrinsics which will be replaced with runtime calls by the
[kit-lower-intrinsics](passes-kit-lower-intrinsics) pass. Yet others, notably
the GPU-centric tapir targets, [cuda](tapir-targets-cuda) and
[hip](tapir-targets-hip), will perform some transformations - specifically,
generating embedded bitcode that will eventually be compiled to GPU
machine-code - but rely on subsequent passes to perform some of the critical
transformations required to produce correct machine code. Clearly, after this
pass has been run, the LLVM module may not be in a state where it can be
handed off to the [back-end](glossary-back-end). This is a fundamental departure
from the core design of LLVM - where every pass is expected to leave the
LLVM module in a "valid" state. This is discussed in greater detail in the
[documentation of Kitsune's pass pipeline](PassPipeline.md).


(passes-loop-stripmine)=
#### loop-stripmine

Performs a blocking transformation on tapir loops. This performs a
transformation similar to the one shown below

```c++
for (i = 0; i < N; i += 1)            for (i = 0; i < N; i += M)
    ...                                   for (j = i; j < i + M; j += 1)
                                              ...
```

In the example above, the outer `for` loop on both the left and right is a
parallel loop. The inner loop on the right is a sequential loop.


(passes-serialize-small-tasks)=
#### serialize-small-tasks

This pass forces "small" tapir tasks to execute in serial. A tapir task is
"small" if the work that it does is deemed to be insufficient to overcome the
overhead of spawning it to run in a parallel execution thread.


(passes-tapir2target)=
#### tapir2target

TODO: Write documentation for this pass.


(passes-tapir-indvars)=
#### tapir-indvars

This runs LLVM's standard
[indvars](https://llvm.org/docs/Passes.html#indvars-canonicalize-induction-variables)
pass, which canonicalizes induction variables, on just the
[tapir loops](glossary-tapir-loop) in a function. It also forces the `indvars`
pass to widen the induction variables on the tapir loops.


(passes-task-canonicalize)=
#### task-canonicalize

This pass canonicalizes tapir tasks so subsequent passes in the pipeline do
not have to perform complicated pattern matching. Currently, all it does is
to split basic blocks in a function at calls to Tapir's `taskframe.create`
intrinsic.


(passes-task-simplify)=
#### task-simplify

This pass simplifies tapir tasks. This includes, among other things, removing
redundant [sync instructions](instructions-sync), removing redundant
[sync regions](glossary-sync-region), simplifying the control-flow graph, and
removing dead calls to Tapir-specific intrinsics.
