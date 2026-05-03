# Tapir Pass Reference

This provides a high-level summary of the [analysis](glossary-analysis-pass),
[transformation](glossary-transformation-pass) and
[meta-passes](glossary-meta-pass) that are a core part of Tapir. These were not
developed specifically for Kitsune, unlike the
[Kitsune-specific passes](KitPassesDoc).

(passes-tapir-analysis)=
## Analysis Passes

This section describes Tapir's analysis passes.


(passes-tasks)=
### tasks

This pass computes and provides information about the
[tapir tasks](glossary-tapir-task) in a function.


(passes-tapir-meta)=
## Meta-Passes

This section describes Tapir's meta-passes i.e. a pass that executes a
specific sequence of passes.


(passes-tapir-lowering)=
### tapir-lowering

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
### tapir-lowering-loops

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
## Transformation Passes

This section describes Tapir's transformation passes.


(passes-loop-spawning)=
### loop-spawning

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
### loop-stripmine

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
### serialize-small-tasks

This pass forces "small" tapir tasks to execute in serial. A tapir task is
"small" if the work that it does is deemed to be insufficient to overcome the
overhead of spawning it to run in a parallel execution thread.


(passes-tapir2target)=
### tapir2target

TODO: Write documentation for this pass.


(passes-tapir-indvars)=
### tapir-indvars

This runs LLVM's standard
[indvars](https://llvm.org/docs/Passes.html#indvars-canonicalize-induction-variables)
pass, which canonicalizes induction variables, on just the
[tapir loops](glossary-tapir-loop) in a function. It also forces the `indvars`
pass to widen the induction variables on the tapir loops.


(passes-task-canonicalize)=
### task-canonicalize

This pass canonicalizes tapir tasks so subsequent passes in the pipeline do
not have to perform complicated pattern matching. Currently, all it does is
to split basic blocks in a function at calls to Tapir's `taskframe.create`
intrinsic.


(passes-task-simplify)=
### task-simplify

This pass simplifies tapir tasks. This includes, among other things, removing
redundant [sync instructions](instructions-sync), removing redundant
[sync regions](glossary-sync-region), simplifying the control-flow graph, and
removing dead calls to Tapir-specific intrinsics.
