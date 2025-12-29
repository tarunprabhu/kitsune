---
orphan: true
---

# Glossary

This is a quick reference to some terminology used in Kitsune and LLVM. This is
not intended to be comprehensive. The primary focus on terms that are unique to
Kitsune. Terms from LLVM that are closely related to Kitsune-specific terms are
also included. Finally, this includes terminology that is not strictly
Kitsune-specific, but is used, perhaps exclusively in Kitsune's documentation.

<!----------------------------------------------------------------------------->

(glossary-a)=
## A

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
  execution unit is generated.

<!----------------------------------------------------------------------------->

(glossary-i)=
## I

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

<!----------------------------------------------------------------------------->

(glossary-l)=

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
**meta pass**
: A name that can be passed, via the `-passes` option to `opt`. However, this
  results in a sequence of
  [passes](glossary-pass) being run, not a single pass. For instance,
  `tapir-lowering` is a meta-pass that runs the passes to lower a
  [tapir loop](glossary-tapir-loop). `kitsune-lowering` is a meta-pass that
  runs both the `tapir-lowering` passes and other, Kitsune-specific passes.

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

<!----------------------------------------------------------------------------->

(glossary-n)=
## N

(glossary-new-pass-manager)=
**new pass manager**
: The current [pass manager](glossary-pass-manager)
  [in LLVM](https://llvm.org/docs/NewPassManager.html) that schedules and
  runs the [middle-end](glossary-middle-end) passes that operate on
  [LLVM-IR](glossary-llvm-ir).

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

<!----------------------------------------------------------------------------->

(glossary-p)=
## P

(glossary-pass)=
**pass**
: In LLVM, this is a unit of code that operates on an
  [IR](glossary-intermediate-representation). The most general pass operates on
  an LLVM [module](glossary-module). Others operate on smaller units contained
  within a module such as functions and loops. Passes
  [may](glossary-transformation-pass) or [may not](glossary-analysis-pass)
  modify the unit of IR on which they operate.

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

(glossay-s)=
## S

(glossary-source-language)=
**source language**
: The high-level language typically used by programmers. This is used to refer
  the languages that can be parsed by Kitsune's [frontends](glossary-front-end).
  Currently, the only source languages that Kitsune officially supports are
  C, C++ and Fortran.

<!----------------------------------------------------------------------------->

(glossary-t)=
## T

(glossary-tapir-instructions)=
**tapir instruction**
: An [LLVM instruction](https://llvm.org/doxygen/classllvm_1_1Instruction.html)
  that is at the core of the
  [tapir extensions to LLVM](https://dl.acm.org/doi/10.1145/3365655).

(glossary-tapir-loop)=
**tapir loop**
: An [LLVM loop](https://llvm.org/doxygen/classllvm_1_1Loop.html) that contains
  tapir instructions that indicate that the iterations of the loop can be
  safely executed independently of one another.

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
