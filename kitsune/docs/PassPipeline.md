# Pass Pipeline and Extension Points

Kitsune has a custom [middle-end](glossary-middle-end)
[pass pipeline](glossary-pass-pipeline) that differs from the standard LLVM
pipeline. This document describes this pipeline and the
[extension points](glossary-extension-point) that are available.

TODO: Write some more text describing this.

<!--
## Preliminaries

Kitsune's [drivers](glossary-driver) run a sequence of [passes](glossary-pass)
on the code being compiled as part of the standard optimization process. This
sequence is very similar, but not identical, to LLVM's standard optimization
[pass pipeline](glossary-pass-pipeline). One of the crucial passes in Kitsune's
pipeline transforms [tapir loops](glossary-tapir-loop) using the
[tapir target](glossary-tapir-target) specified by the user. This is the
[loop-spawning](passes-loop-spawning) pass, but it requires a number of other passes, particularly
[loop-simplify](https://llvm.org/docs/Passes.html#loop-simplify-canonicalize-natural-loops)
to have been run on the tapir loops first.
-->
