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

(passes-kitsune-analysis)=
### Analysis Passes

(passes-tapir-target-analysis)=
#### tapir-target-analysis


(passes-kitsune-meta)=
### Meta-Passes

(passes-kit-lowering)=
#### kit-lowering


(passes-kitsune-transformation)=
### Transformation Passes

(passes-emb-link-libdevice-bitcode)=
#### emb-link-libdevice-bitcode

(passes-emb-optimize)=
#### emb-optimize

(passes-emb-prepare)=
#### emb-prepare

(passes-resolve-libdevice-calls)=
#### emb-resolve-libdevice-calls

(passes-kit-cgfb)=
#### kit-cgfb

(passes-kit-ctors)=
#### kit-ctors

(passes-kit-kernel-properties)=
#### kit-kernel-properties

(passes-kit-prefetch)=
#### kit-prefetch

(passes-kit-lower-intrinsics)=
#### kit-lower-intrinsics

(passes-kit-strip-addr-spaces)=
#### kit-strip-addr-spaces

<!----------------------------------------------------------------------------->

(passes-tapir)=
## Tapir Passes

(passes-tapir-analysis)=
### Analysis Passes

(passes-tasks)=
#### tasks

(passes-tapir-meta)=
### Meta-Passes

(passes-tapir-lowering)=
#### tapir-lowering

(passes-tapir-transformation)=
### Transformation Passes

(passes-loop-spawning)=
#### loop-spawning

(passes-loop-stripmine)=
#### loop-stripmine

(passes-serialize-small-tasks)=
#### serialize-small-tasks

(passes-tapir2target)=
#### tapir2target

(passes-tapir-indvars)=
#### tapir-indvars

(passes-task-canonicalize)=
#### task-canonicalize

(passes-task-simplify)=
#### task-simplify
