---
orphan: true
---

# Glossary

This is a quick reference to some terminology used in Kitsune and LLVM. This is
not intended to be comprehensive. The primary focus is on terms that are unique
to Kitsune. Terms from LLVM that are closely related to Kitsune-specific terms
are also included. Finally, this includes terminology that is not strictly
Kitsune-specific, but is used, perhaps exclusively, in Kitsune's documentation.

<!----------------------------------------------------------------------------->

(glossary-a)=
## A

(glossary-analysis-manager)=
**analysis manager**
: An object that works in conjunction with a
  [pass manager](glossary-pass-manager) to run
  [analysis passes](glossary-analysis-pass) as needed, cache the results and
  make them available to other [passes](glossary-pass). If a pass invalidates an
  analysis, the analysis manager will recompute the analysis on demand. This
  improves the performance of the compiler by reducing the number of times
  analyses are computed.

(glossary-analysis-pass)=
**analysis pass**
: A [pass](glossary-pass) that does not modify the
  [IR](glossary-intermediate-representation) on which it operates.
  An analysis pass determines properties of the IR that can be used by
  [transformation passes](glossary-transformation-pass). For instance,
  the
  [alias analysis pass](https://llvm.org/docs/Passes.html#basic-aa-basic-alias-analysis-stateless-aa-impl)
  is an analysis pass that is used by the
  [vectorization passes](https://llvm.org/docs/Vectorizers.html)
  to check if it is safe to vectorize.

(glossary-asynchronous-intrinsic)=
**asynchronous intrinsic**
: Same as a [non-blocking intrinsic](glossary-non-blocking-intrinsic).

<!----------------------------------------------------------------------------->

(glossary-b)=
## B

(glossary-back-end)=
**back-end**
: The part of the compiler that generates machine code. In LLVM, this generates
  machine code from [LLVM-IR](glossary-llvm-ir).

(glossary-backend)=
**backend**
: See [back-end](glossary-back-end)

(glossary-bitcode)=
**bitcode**
: The binary serialization of [LLVM-IR](glossary-llvm-ir). These are typically
  saved to files with a `.bc` extension.

<!----------------------------------------------------------------------------->

(glossary-c)=
## C

(glossary-call-site)=
**call site**
: The location where a function is called.

(glossary-callsite)=
**callsite**
: Same as [call site](glossary-call-site).

(glossary-cgscc-pass)=
**cgscc pass**
: A [pass](glossary-pass) that traverses the callgraph of a
  [module](glossary-module) from the bottom-up (callees before callers).
  These are typically analysis passes, but they need not be. They must also
  satisfy the criteria enumerated
  [here](https://llvm.org/docs/WritingAnLLVMPass.html#the-callgraphsccpass-class).
  In most cases, a [function pass](glossary-function-pass) should probably be
  preferred over writing a CallGraphSCC pass.

(glossary-codegen)=
**codegen**
: The process of generating machine code from some
  [IR](glossary-intermediate-representation).
  This is performed by Kitsune's [back-end](glossary-back-end). Within some
  LLVM projects, this term is used to mean something else - for instance, in
  `clang`, the term 'codegen' refers to the process of generating
  [LLVM-IR](glossary-llvm-ir) from the
  [AST](https://clang.llvm.org/docs/IntroductionToTheClangAST.html).
  In Kitsune, we only use it to mean machine-code generation.

<!----------------------------------------------------------------------------->

(glossary-d)=
## D

(glossary-dependent-pass)=
**dependent pass**
: A pass that requires one or more "[requirable](glossary-requirable-pass)"
  passes to have run before it is run.

(glossary-device)=
**device**
: In the context of GPU-centric [tapir targets](glossary-tapir-target), the
  device typically refers to a GPU. In principle, it could be used to refer to
  any accelerator, though, at the time of writing, Kitsune only supports GPUs
  as accelerators.

(glossary-device-code)=
**device code**
: Machine code for an accelerator, typically a GPU. This term is usually used in
  the context of "embedded device code". This is where a code generation pass
  generates device code and sets it as the initializer of a global variable
  in [LLVM-IR](glossary-llvm-ir). When this is compiled to an object file, the
  device code will be "embedded" in the [host](glossary-host) machine code.
  The name of the global variable will become the name of a symbol that can be
  used by Kitsune's runtime to retrieve this device code.

(glossary-device-function)=
**device function**
: A function that can only run on an accelerator - typically a GPU.
  Device functions can only be called by
  [kernel functions](glossary-kernel-function), or by other device functions.
  They are almost always
  private to a [device module](glossary-device-module).

(glossary-device-module)=
**device module**
: An LLVM [module](glossary-module) from which accelerator code is generated.
  An accelerator here is any execution unit that is _not_ the primary execution
  unit. The primary execution unit is typically the CPU on which a program is
  launched. Currently, the only accelerators that are supported are NVIDIA and
  AMD GPU's.

(glossary-driver)=
**driver**
: The language-specific command-line utility used to compile
  [source](glossary-source-language) code. Kitsune currently provides drivers
  for C, C++ and Fortran.
: The driver also refers to the part of the [front-end](glossary-front-end)
  that parses the command-line options before calling the language-specific
  compiler - `cc1` for C and C++, `fc1` for Fortran.

<!----------------------------------------------------------------------------->

(glossary-e)=
## E

(glossary-embedded-bitcode)=
**embedded bitcode**
: The serialized [LLVM bitcode](glossary-bitcode) representation of a
  [device module](glossary-device-module) that is stored as the initializer of
  a global variable in a [host module](glossary-host-module).
  [This document](EmbeddedBitcode.md) contains more information about the
  design, implementation and use of embedded bitcode in Kitsune.

(glossary-embedded-bitcode-pass)=
**embedded bitcode pass**
: Unlike other [LLVM passes](glossary-pass), these are passes that operate
  exclusively on embedded bitcode. The only change that they will make in the
  [host module](glossary-host-module) is a change to the initializer of the
  global variable containing the embedded bitcode.
  [This document](EmbeddedBitcode.md) contains more information about the
  design, implementation and use of embedded bitcode in Kitsune.

(glossary-embedded-module)=
**embedded module**
: The [module](glossary-module) obtained by deserializing the
  [embedded bitcode](glossary-embedded-bitcode) found in a
  [host module](glossary-host-module).

(glossary-extension-point)=
**extension point**
: Locations in a [pass pipeline](glossary-pass-pipeline) at which passes
  dynamically loaded from a [pass plugin](glossary-pass-plugin) can be
  scheduled. LLVM provides a number of extension points that allow passes to be
  added to the standard pass pipeline. Kitsune provides additional extension
  points intended for [embedded bitcode passes](glossary-embedded-bitcode-pass).

<!----------------------------------------------------------------------------->

(glossary-f)=
## F

(glossary-front-end)=
**front-end**
: The part of the compiler that is responsible for initial process of a
  [source language](glossary-source-language). In Kitsune's documentation, we
  deem the part of the compiler that emits some
  [intermediate representation](glossary-intermediate-representation) such as
  [LLVM-IR](glossary-llvm-ir) or MLIR to be a frontend.
: The term is also sometimes used to refer to Kitsune's language-specific
  [drivers](glossary-driver).

(glossary-frontend)=
**frontend**:
: See [frontend-end](glossary-front-end)

(glossary-function-analysis-manager)=
**function analysis manager**
: An [analysis manager](glossary-analysis-manager) specifically for
  [function analyses](glossary-function-analysis-pass).

(glossary-function-analysis-pass)=
**function analysis pass**
: An [analysis pass](glossary-analysis-pass) that computes properties of a
  function.

(glossary-function-pass)=
**function pass**
: A [pass](glossary-pass) whose unit is an LLVM function. Such passes may make
  changes to a function by adding or removing instructions and basic blocks,
  but may not make add/remove functions from the [module](glossary-module) in
  which they are contained.

<!----------------------------------------------------------------------------->

(glossary-g)=
## G

(glossary-grain-size)=
**grain size**
: TODO: Explain how grain size works. Some tapir targets ignore it altogether,
  others always set it to 1, etc.

<!----------------------------------------------------------------------------->

(glossary-h)=
## H

(glossary-host)=
**host**
: In GPU-centric [tapir targets](glossary-tapir-target), the host typically
  refers to the primary execution unit, that is, the CPU.

(glossary-host-module)=
**host module**
: The LLVM [module](glossary-module) from which code that will run on the main
  execution unit is generated. The main execution unit is nearly always a
  general-purpose CPU.

<!----------------------------------------------------------------------------->

(glossary-i)=
## I

(glossary-inlining)=
**inlining**
: A transformation performed by most compilers that replaces a function
  [call site](glossary-call-site) with the body of the called function. See
  also: [outlining](glossary-outlining).

(glossary-intermediate-representation)=
**intermediate representation**
: Commonly abbreviated IR, this is the code used internally by a compiler to
  represent source code. LLVM, and by extension, Kitsune, uses several
  IR's. The best known of these is [LLVM-IR](glossary-llvm-ir) - the
  intermediate representation used in LLVM's [middle-end](glossary-middle-end).
  Others include
  [MIR](https://llvm.org/docs/CodeGenerator.html#machine-code-representation)
  used in the [back-end](glossary-back-end) and
  [MLIR](https://mlir.llvm.org/docs/Dialects/).

(glossary-in-tree-pass)=
**in-tree pass**
: A [pass](glossary-pass) that is part of LLVM's source code. This term is
  nearly always used to refer to a pass that has been developed "downstream"
  of LLVM i.e. in projects that are not part of LLVM but that build on top of
  it, such as Kitsune. For instance,
  [loop-spawning](passes-loop-spawning) is an in-tree pass. These passes are
  built when LLVM is built.

<!----------------------------------------------------------------------------->

(glossary-k)=
## K

(glossary-kernel-function)=
**kernel function**
: A function that can only run on an accelerator - typically a GPU. They can
  only be called from a function that is already executing on the
  [host](glossary-host). In other words, they are "launched" by the host to
  run on a [device](glossary-device). They can neither be called by
  [device functions](glossary-device-function), nor by other kernel functions.
  They are usually generated from the body of a
  [tapir loop](glossary-tapir-loop).

(glossary-kernel-module)=
**kernel module**
: Synonym for a [device module](glossary-device-module).

(glossary-l)=
## L

(glossary-legacy-pass)=
**legacy pass**
: A [pass](glossary-pass) that is managed by the
  [legacy pass manager](glossary-legacy-pass-manager).

(glossary-legacy-pass-manager)=
**legacy pass manager**
: The old [pass manager](glossary-pass-manager) in LLVM that has been superseded
  by the [new pass manager](glossary-new-pass-manager) in
  LLVM's [middle-end](glossary-middle-end).
  Currently, this is only used by the [codegen](glossary-codegen)
  [pipeline](glossary-pass-pipeline) in LLVM's [back-end](glossary-back-end).

(glossary-libdevice-bitcode-file)=
**libdevice bitcode file**
: Both NVIDIA and AMD provide definitions of library functions, such as the C
  standard [mathematical functions](https://www.sourceware.org/newlib/libm.html)
  as [bitcode](glossary-bitcode) files that are to be linked into
  [LLVM-IR](glossary-llvm-ir) that is being compiled for the corresponding GPU.
  These files are referred to as "libdevice bitcode files".

(glossary-loop-pass)=
**loop pass**
: A [pass](glossary-pass) whose unit is a loop in a function. These are
  processed in loop-nest order such that the outermost loop is processed last.
  These passes must only modify the loop that they are operating on. Any
  parent loops must not be modified. In many cases, it may be more convenient to
  write a [function pass](glossary-function-pass) instead of a loop pass.

(glossary-lowering)=
**lowering**
: The process of transforming code from a higher-level language to a lower-level
  language. The higher the level of a language, the closer it is to a
  [source language](glossary-source-language) intended for use by programmers.
  The lower the level, the closer the language is to machine code. Assembly
  code is the lowest level language since, unlike machine code, it
  human-readable.
: In Kitsune, we also use the term "lowering" to mean
  [tapir lowering](glossary-tapir-lowering).

(glossary-llvm-assembly)=
**LLVM assembly**
: Human-readable representation of [LLVM-IR](glossary-llvm-ir). These are
  typically saved to files with a `.ll` extension.

(glossary-llvm-ir)=
**LLVM IR**
: LLVM's [middle-end](glossary-middle-end)
  [intermediate representation](https://en.wikipedia.org/wiki/Intermediate_representation).

<!----------------------------------------------------------------------------->

(glossary-m)=
## M

(glossary-meta-pass)=
**meta-pass**
: A name that, when passed to the `-passes` option to `opt` results in a
  sequence of [passes](glossary-pass) being run, not a single pass. For instance,
  [tapir-lowering](passes-tapir-lowering) is a meta-pass that runs the passes to
  lower a [tapir loop](glossary-tapir-loop).
  [kit-lowering](passes-kit-lowering) is a meta-pass that
  runs both the `tapir-lowering` passes and other, Kitsune-specific, passes. One
  can think of a meta-pass as a "named [pass pipeline](glossary-pass-pipeline)".

(glossary-middle-end)=
**middle-end**
: The middle-end of is the part of the compiler that runs
  source-language-independent and
  machine-independent [passes](glossary-pass) on an
  [IR](glossary-intermediate-representation). [LLVM-IR](glossary-llvm-ir) is
  the best-known middle-end IR in LLVM. In Kitsune's documentation, the
  [MLIR](https://mlir.llvm.org/) passes are also considered as part of LLVM's
  middle-end, even though some dialects, such as
  [FIR](https://flang.llvm.org/docs/FIRLangRef.html)
  (Fortran Intermediate Representation) are very
  closely tied to a specific [source language](glossary-source-language).

(glossary-module)=
**module**
: This is the top-level container of all other [LLVM-IR](glossary-llvm-ir)
  objects. A module typically corresponds to a single
  [source](glossary-source-language) file, but they can be built in other ways.

(glossary-module-analysis-manager)=
**module analysis manager**
: An [analysis manager](glossary-analysis-manager) specifically for
  [module analyses](glossary-function-analysis-pass).

(glossary-module-analysis-pass)=
**module analysis pass**
: An [analysis pass](glossary-analysis-pass) that computes properties of a
  [module](glossary-module).

(glossary-module-pass)=
**module pass**
: The most general [pass](glossary-pass) that treats the entire LLVM
  [module](glossary-module) as a unit. Such passes may make changes at all
  levels of the module, from individual instructions in a function, to
  adding/removing global variables.

<!----------------------------------------------------------------------------->

(glossary-n)=
## N

(glossary-new-pass-manager)=
**new pass manager**
: The current [pass manager](glossary-pass-manager)
  [in LLVM](https://llvm.org/docs/NewPassManager.html) that schedules and
  runs the [middle-end](glossary-middle-end) passes that operate on
  [LLVM-IR](glossary-llvm-ir).

(glossary-non-blocking-intrinsic)=
**non-blocking intrinsic**
: An intrinsic function that returns before the operation that it has performed
  has been completed. The operation will continue in the background. Such
  intrinsics will typically a (possibly opaque) context object that can be
  waited on by other intrinsics.

<!----------------------------------------------------------------------------->

(glossary-o)=
## O

(glossary-optimization-pass)=
**optimization pass**
: A [transformation pass](glossary-transformation-pass) whose that is intended
  to improve the performance of the machine code that will eventually be
  generated. All optimization passes are transformation passes, but not all
  transformation passes are optimization passes.

(glossary-optional-tapir-target)=
**optional tapir target**
: A [tapir target](glossary-tapir-target) that can be optionally enabled when
  building Kitsune. A tapir target that is not in the list of
  [universal tapir targets](glossary-universal-tapir-target) is, by definition,
  an optional tapir target.

(glossary-outlining)=
**outlining**
: A transformation where a region of code is replaced with a call to a function.
  The body of this called function consists of the region of code that was
  replaced. See also: [inlining](glossary-inlining).

(glossary-out-of-tree-pass)=
**out-of-tree pass**
: A [pass](glossary-pass) that is not part of LLVM's source code.
  code. This is developed and built entirely independently of Kitsune (or LLVM).
  Such passes are almost always part of a [pass plugin](glossary-pass-plugin)
  that can be dynamically loaded and used by both LLVM's (and Kitsune's)
  frontends as well as LLVM tools such as
  [opt](https://llvm.org/docs/CommandGuild/opt.html).

<!----------------------------------------------------------------------------->

(glossary-p)=
## P

(glossary-pass)=
**pass**
: In LLVM, this is a unit of code that operates on a unit of
  [IR](glossary-intermediate-representation). Depending on the IR unit on which
  a pass operates, it may be characterized as a
  [module pass](glossary-module-pass), [function pass](glossary-function-pass),
  [loop pass](glossary-loop-pass), or a [CGSCC pass](glossary-cgscc-pass).
  Passes [may](glossary-transformation-pass) or
  [may not](glossary-analysis-pass) modify the unit of IR on which they operate.

(glossary-pass-manager)=
**pass manager**
: In LLVM, a pass manager schedules passes to run on some
  [IR](glossary-intermediate-representation). It also manages
  [analysis passes](glossary-analysis-pass) and ensures that the results that
  they provide are up-to-date when they are requested by
  [transformation passes](glossary-transformation-pass). LLVM currently contains
  two pass managers, a [legacy pass manager](glossary-legacy-pass-manager) and
  a [new pass manager](glossary-new-pass-manager).

(glossary-pass-pipeline)=
**pass pipeline**
: A sequence of [passes](glossary-pass) added to a
  [pass manager](glossary-pass-manager). The "O0 pass pipeline", for instance,
  refers to the sequence of passes that are run when the `-O0` command-line
  option is provided to either a [driver](glossary-driver) or an LLVM tool such
  as `opt`. A [meta pass](glossary-meta-pass) can be thought of as a "named"
  pass pipeline that can be explicitly run using `opt`.

(glossary-pass-plugin)=
**pass plugin**
: A dynamic shared object containing [LLVM-IR](glossary-llvm-ir)
  [passes](glossary-pass). A pass plugin can be used to add passes - developed
  outside Kitsune - to the [pass pipeline](glossary-pass-pipeline) at
  [use-time](glossary-use-time).

(glossary-primary-tapir-target)=
**primary tapir target**
: This is the tapir target provided as the value of the `--tapir` command-line
  option that is known to the Kitsune drivers, `kitcc`, `kit++` and `kitfc`
  as well as tools such as
  [opt](https://llvm.org/docs/CommandGuide/opt.html) and
  [llc](https://llvm.org/docs/CommandGuide/llc.html).

(glossary-pseudo-tapir-target)=
**pseudo tapir target**
: A [tapir target](glossary-tapir-target) that does not actually lower a
  [tapir loop](glossary-tapir-loop). This is primarily useful for debugging
  and writing tests. Currently, [nolo](tapir-targets-nolo) is the only such
  tapir target in Kitsune.

<!----------------------------------------------------------------------------->

(glossary-r)=
## R

(glossary-requirable-pass)=
**requirable pass**
: A pass that must be run before a "[dependent pass](glossary-dependent-pass)"
  that requires it is run. These passes will usually add metadata to the IR
  unit on which they operate - typically a function or module - indicating when
  they have been run.

<!----------------------------------------------------------------------------->

(glossay-s)=
## S

(glossary-separate-compilation)=
**separate compilation**
: When compiling a project consisting of many source files, this refers to the
  technique of compiling each source file into a corresponding object file
  first, then linking the object files to form the final executable (or
  dynamic shared object). This is the standard technique used in most real
  world code, especially when using build systems such as
  [CMake](https://cmake.org) or [Meson](https://mesonbuild.com/). The
  alternative approach would be to pass all the source files to the compiler
  in one single invocation. This is occasionally seen in hand-written build
  systems, and is sometimes used when compiling very small projects by hand.

(glossary-source-language)=
**source language**
: The high-level language typically used by programmers. This is used to refer
  the languages that can be parsed by Kitsune's [frontends](glossary-front-end).
  Currently, the only source languages that Kitsune officially supports are
  C, C++ and Fortran.

(glossary-sync-region)=
**sync region**
: A sync region is a "tag" associated with [tapir tasks](glossary-tapir-task).
  All [tapir instructions](glossary-tapir-instruction) take a sync region as an
  operand. This "tag" is obtained by calling tapir's `llvm.syncregion.start`
  intrinsic.

<!----------------------------------------------------------------------------->

(glossary-t)=
## T

(glossary-tapir-instruction)=
**tapir instruction**
: An [LLVM instruction](https://llvm.org/doxygen/classllvm_1_1Instruction.html)
  that is at the core of the
  [tapir extensions to LLVM](https://dl.acm.org/doi/10.1145/3365655). The list
  of tapir instructions can be found [here](instructions-tapir).

(glossary-tapir-loop)=
**tapir loop**
: An [LLVM loop](https://llvm.org/doxygen/classllvm_1_1Loop.html) whose body is
  bounded by a Tapir [detach](instructions-detach) and
  [reattach](instructions-reattach) instruction. Every iteration of such a loop
  can be safely executed independently of every other iteration.

(glossary-tapir-lowering)=
**tapir-lowering**
: The transformation of a [tapir loop](glossary-tapir-loop) to use a runtime
  system, or to a form suitable for execution on a [device](glossary-device).

(glossary-tapir-target)=
**tapir target**
: The object that transforms a [tapir loop](glossary-tapir-loop). The
  GPU-centric tapir targets, [cuda](tapir-targets-cuda) and
  [hip](tapir-targets-hip), transform the loop into a form expected to
  execute efficiently on a GPU. Others, such as
  [opencilk](tapir-targets-opencilk)
  insert appropriate API calls, in this case to the
  [OpenCilk runtime](https://github.com/OpenCilk/cheetah). For more information,
  see the [general overview of Kitsune](Overview.md) and the document describing
  the [supported tapir targets](TapirTargets.md).

(glossary-tapir-task)=
**tapir task**
: A tapir task is a region of code bounded by Tapir's.
  [detach](instructions-detach) and [reattach](instructions-reattach)
  instructions. This code can, in principle, be run in parallel with other code.
  All tapir tasks are associated with a [sync region](glossary-sync-region).

(glossary-transformation-pass)=
**transformation pass**
: A [pass](glossary-pass) that transforms the
  [IR](glossary-intermediate-representation) in some way. In most cases, these
  transformations are also optimizations i.e. they are intended to improve the
  performance of the machine code that will eventually be generated. However,
  some transformation passes do not directly improve performance. For instance,
  the
  [function attributor pass](https://llvm.org/docs/Passes.html#function-attrs-deduce-function-attributes)
  is an inter-procedural pass that computes function attributes. Other
  transformation passes can use these attributes to carry out more aggressive
  optimizations.

<!----------------------------------------------------------------------------->

(glossary-u)=
## U

(glossary-universal-tapir-target)=
**universal tapir target**
: A [tapir target](glossary-tapir-target) that is always enabled when Kitsune is
  built. This is the current list of universal tapir target:
  {{kitsune_guaranteed_tapir_targets_list}}

(glossary-use-time)=
**use-time**
: We use the term use-time to indicate when Kitsune is used to compile
  [user code](glossary-user-code). This is used when the standard terms
  "compile-time" and "run-time" cannot be used unambiguously.

(glossary-user-code)=
**user code**
: Code that is (usually in the process of being) compiled by Kitsune
