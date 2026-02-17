# FAQ

The questions here have been raised on several occasions, so we address them
here.


(faq-difference-kitsune-tapir)=
## What is the difference between Kitsune and Tapir?

Tapir is an extension to LLVM that allows fork-join parallelism to be exposed
to LLVM's optimizer. This extension adds some
[instructions](KitInstructionsDoc.md) and [passes](KitPassesDoc.md) to LLVM.
It also includes modifications to the
[standard LLVM passes](https://llvm.org/docs/Passes.html) to work correctly with
these instructions. Kitsune builds on these extensions and adds a number of
[tapir targets](glossary-tapir-target) for, most notably, NVIDIA
([cuda](tapir-targets-cuda)) and AMD ([hip](tapir-targets-hip)) GPUs and also
other runtime systems ([pthreads](tapir-targets-pthreads)) [^1]. Kitsune's
[language extensions](LanguageExtensions.md) are also entirely distinct.

Tapir was developed by
[Schardl et al](https://dl.acm.org/doi/10.1145/3365655). This forms the core
of the [OpenCilk](https://www.opencilk.org/)
[compiler](https://github.com/OpenCilk/opencilk-project). Kitsune originally
started out as a fork of the OpenCilk compiler. Over time, Kitsune has diverged
from that code base, primarily by removing the OpenCilk-specific additions from
the [front-end](glossary-front-end), and later by making some changes to the
[middle-end](glossary-middle-end). Kitsune's command-line options will often
begin with `--tapir-`, particularly when the option pertains to a tapir target.

[^1]: A Tapir target for [Realm](https://legion.stanford.edu/tutorial/realm/events_basic.html) has also been developed, but is not supported. More details can be found [here](unsupported-tapir-targets).


(faq-terminology-consistency)=
## Why is the terminology in the source-level documentation not always consistent with the HTML documentation?

The original architecture of Kitsune bears little resemblance to its current
form. Some of the in-source comments, variable names etc. may still reflect the
older form. For instance, you may see references to "backends" which really
mean "tapir target". The documentation that you are currently reading was
written some years after development on Kitsune began, and after a major
redesign. We made a conscious effort to be consistent with terminology when
writing it, but we have not gone back over the code base and changed the inline
comments and variable names to be consistent with this external documentation.
That is being done, but it will take some time before it is completed.


(faq-compiler-id)=
## Why does Kitsune identify itself as Clang?

This is most noticeable when using Kitsune with [CMake](https://cmake.org). When
setting `CMAKE_C_COMPILER` or `CMAKE_CXX_COMPILER` to the appropriate Kitsune
driver, the output will include the lines below

```
-- The C compiler identification is Clang 21.1.3
-- The CXX compiler identification is Clang 21.1.3
```

One might reasonably expect the output to be something close to the following

```
-- The C compiler identification is Kitsune 0.21.0
-- The CXX compiler identification is Kitsune 0.21.0
```

If Kitsune were to identify itself correctly, `cmake` would likely have to be
patched to ensure that it generated the correct command line options to be used
with Kitsune. Kitsune is primarily a research compiler. While being built on
LLVM provides us with a robust base, the primary objective is to push the limits
of what compilers can do - it is not to deliver and maintain a product that can
be used in a wide range of scenarios. Since Kitsune's C and C++ drivers accepts
all the relevant command-line options that `Clang` does, identifying ourselves
as "Clang" allows us to easily compile codes that use `cmake` without requiring
us to also patch `cmake`.


(faq-compiler-version)=
## What version of Kitsune am I actually using?

Since Kitsune is research prototype, there have been no official "releases" as
such, so there are no official versions either. We regularly rebase Kitsune on
LLVM's releases (usually within a few weeks of a new major release). To identify
which release of LLVM Kitsune has been built upon, use Kitsune's
the [kit-config](CommandGuide/kit-config.md) utility as shown below.

```
kit-config --version
LLVM version: 21.1.3
Kitsune version: 0.21.0
```

The last line of the output contains Kitsune's "version". The minor version here
will always be the major version of the LLVM release on which Kitsune has been
built.


(faq-language-support)=
## Can Kitsune be used with other languages?

We have shown how this could be done in limited cases with languages such as
Python and Pascal. These were mostly with toy examples and were intended to help
us assess how it was feasible to do so. These demos have not been made public.
There are no plans to officially support any other language at this time.
