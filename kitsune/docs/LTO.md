# Link-time Optimization (LTO)

This describes Kitsune's use of
[link-time optimization](https://llvm.org/docs/LinkTimeOptimization.html) (LTO)
and how support for it has been implemented. LTO refers to
inter-[module](glossary-module) optimizations that take place at link-time.

## Preliminaries

Before describing Kitsune's design, we describe some subtleties of compiling
with LTO enabled that might not be immediately obvious. Those familiar with the
intricacies of LLVM's LTO implementation may safely skip this section.

### Bitcode Files, not Object Files

To enable LTO, the `-flto` command-line option must be used when compiling.
In the example below, several files are being compiled simultaneously to
produce the final executable `a.out`.

```shell
kit++ -O2 -flto -o a.out in1.cpp in2.cpp in3.cpp
```

From the user's perspective, the compiler's behavior will be the same as it
would have been had `-flto` not been provided. Depending on the contents of the
[source](glossary-source-language) files, compiling with LTO enabled may take
longer since optimizations will be run at both compile-time and link-time.

```{note}
In the example above, we have not specified a
[tapir target](glossary-tapir-target) for simplicity.
Kitsune's [drivers](glossary-driver) can be used without a tapir target in
which case, it is mostly equivalent to using `clang` [^1].
```

[^1]: Command-line options for OpenCL and other languages that are not supported by Kitsune, but are supported by `clang` will probably not work in this case.

However, when generating object files, the compiler behaves differently when
LTO has been enabled. In the example below, a single source file is being
compiled to an object file.

```shell
kit++ -O2 -flto -c -o in.o in.cpp
```

What follows is the result of giving the object file that was just generated to
the POSIX standard
[file](https://pubs.opengroup.org/onlinepubs/000095399/utilities/file.html)
utility.

```shell
file in.o
in.o: LLVM IR bitcode
```

Normally, one would have to add the `-emit-llvm` option to have the compiler
generate [bitcode](glossary-bitcode). However, when LTO is enabled, generating
an object file implicitly generates bitcode instead. When linking the object
files that were generated, one would now be linking LLVM bitcode instead. While
this might seem odd, if one is only interested in compiling and linking with
LTO enabled, and doesn't care about the formats of the intermediate products,
the process works exactly as might be expected. In the example below, the
executable `a.out` is generated with LTO enabled, but this time, by separately
compiling each input file.

```
kit++ -O2 -flto -c -o in1.o in1.cpp
kit++ -O2 -flto -c -o in2.o in2.cpp
kit++ -O2 -flto -c -o in3.o in3.cpp
kit++ -O2 -flto -o a.out in1.o in2.o in3.o
```

The final `a.out` file produced here would be exactly equivalent to the file
produced by compiling all input files in a single command.

### Linking LLVM Bitcode

As described in the previous section, carrying out
[separate compilation](glossary-separate-compilation) with LTO enabled results
in LLVM bitcode files being generated, not object files as one would expect.
At link-time, then, the linker would have know how to link bitcode. Depending
on the linker used, this is accomplished in different ways.

#### System Linker

By default, Kitsune uses the system's linker. On Linux and FreeBSD, this is
usually the [GNU linker](https://sourceware.org/binutils/docs/ld/). Kitsune's
[drivers](glossary-driver), like `clang` and `flang` will automatically invoke
this linker when appropriate. To examine this linker invocation, we use the
`-###` command-line option as shown below.

```shell
kit++ -### -O2 -flto -o a.out in1.o in2.o in3.o
```

This produces a fairly lengthy command-line invocation of ld. We present only
the most relevant parts of the output below. For clarity, the output has been
split across several lines.

```shell
ld -o a.out \
   -plugin /home/tarun/workspace/kitsune/build/bin/../lib/LLVMgold.so \
   -plugin-opt=O2 \
   in1.o in2.o in3.o \
    ...
```

Note the use of a linker plugin here, in this case LLVM's
[gold](https://llvm.org/docs/GoldPlugin.html) plugin [^2]. This plugin is
capable of handling LLVM bitcode. Note that the optimization
level is explicitly passed to the plugin via the `-plugin-opt`  command-line
option. With this, the plugin constructs the appropriate optimization pipeline
that can operate on the provided "object" files that are actually consist of
[LLVM-IR](glossary-llvm-ir).

[^2]: This was originally developed for the [gold linker](https://en.wikipedia.org/wiki/Gold_(linker)) at a time when the GNU linker, `ld`, did not support plugins. Since then, support of plugins has been added to the GNU linker as well. As of binutils 2.44, the gold linker [has been deprecated](https://sourceware.org/binutils/docs/ld/Plugins.html).

Since Kitsune's pass pipeline is only enabled when a
[tapir target](glossary-tapir-target) has been explicitly specified
(see [here](PassPipeline.md) for more details), to use this plugin for LTO
support in Kitsune, the gold plugin would have to be modified to accept at
least the `--tapir` command-line option, and probably several others as well.

#### LLD

Instead of relying on the system's linker and plugins to support linking LLVM
bitcode, LLVM's own linker, [lld](https://lld.llvm.org/) could be used instead.
To do this, the `-fuse-ld=lld` command-line option should be used.

```shell
kit++ -### -O2 -flto -fuse-ld=lld -o a.out in1.o in2.o in3.o
```

In this case, the linker invocation will be different. A simplified invocation,
reformatted for clarity is shown below.

```shell
ld.lld -o a.out \
       -plugin-opt=O2 \
       in1.o in2.o in3.o \
       ...
```

Note that here, the `-plugin` option is absent. This is because `lld` has
built-in support to handle LLVM bitcode as well as object files without
requiring a plugin. The -O2 option is, nevertheless, passed to LLD via the
`-plugin-opt` command-line option, presumably because it is recognized as a
"compiler" option, not an option that is native to `lld`.

```{tip}
In this particular case, it may actually be better to provide an optimization
level to lld directly as shown below. Note that we must explicitly state that
`--lto-O2` is a linker option by adding `-Xlinker` before it.

````shell
kit++ -Xlinker --lto-O2 -flto -fuse-ld=lld -o a.out in1.o in2.o in3.o
```

Once again, to enable Kitsune's pass pipeline when linking with LLD, the
`--tapir` and other command-line options would have to be passed to it.

## Kitsune's Implementation

Kitsune has full support for LTO. The highlights of the implementation are as
follows:

<!--
     XXX:
     We deliberately use a . in the first bullet and a ) in the second because
     it seems to be the only way to force a space to be added between the two
     bullets in the rendered HTML. Yes, it is utterly stupid, but I can't seem
     to find a way of doing this "properly", despite the documentation telling
     me that leaving an empty line between the bullet points in the markdown
     "will work"
-->
1. LLD is required for LTO when a tapir target is specified. Other linkers are
   not supported. Note that other linkers _**are**_ supported when linking
   without LTO.

2) The `--tapir` and other command-line options are *not* exposed to LLD.
   Instead, they are passed as "LLVM" options. LLD, like `clang` and `flang`
   allows direct access to the
   [command-line options](https://llvm.org/docs/CommandLine.html)
   defined within LLVM passes and other libraries via the
   `-mllvm` command-line option.

As discussed [here](CodeOrganization.md), one of the design choices that we
made for Kitsune was to avoid making changes to the "core" LLVM projects
(Clang, Flang, LLD, LLVM, and so on) when possible. This was to
make the task of keeping up with LLVM easier. Since LLD was already capable of
linking LLVM bitcode files, it seemed redundant to maintain support for
Kitsune-specific command-line options in the gold plugin. It would also
eliminate the maintenance burden of keeping the changes to the plugin up-to-date
with LLVM.

As discussed [here](CommandLineOptions.md), Kitsune exposes the same set of
(Kitsune-specific) command-line options to both the drivers and the LLVM tools.
This was a deliberate choice to allow us to use the driver and
[LLVM's tools](LLVMTools.md) with as little friction as possible.
This also means that the these command-line options are available to _all_
LLVM tools. While
we could have added Kitsune-specific command-line options as first class
entities to LLD, we would have had to do so for each of the "backends" that both
LLD and Kitsune support (currently, `ELF` and `MachO`, though this may expand
in the future). Kitsune's command-line options are already duplicated between
the drivers and the LLVM tools. Adding more sets of command-line options
that must be kept consistent would require more engineering and
maintenance effort. Instead, we chose to have Kitsune's
[driver](DriverDesign.md) pass the Kitsune-specific command-line options to LLD
as "internal LLVM" options.
It essentially allows us to get Kitsune-specific command-line options into _all_ LLD backends "for free". As a result, the number of changes that had to be made
to LLD to support Kitsune and LTO were very minimal.

The command line below provides the serial tapir target when linking with LTO
enabled (assume that the same tapir target was provided when compiling the
"object files" used here)

```shell
kit++ -### -O2 -flto --tapir=serial -o a.out in1.o in2.o in3.o
```

The relevant part of the linker invocation generated by Kitsune's driver in this
case is shown here.

```shell
ld.lld -o /dev/null \
       -plugin-opt=O2 \
       in1.o in2.o in3.o \
       --lto-O2 \
       -mllvm --tapir=serial \
       ...
```

Note how the `--tapir=serial` option is passed using `-mllvm`. Note also that
the optimization level is automatically added using the corresponding
LTO-specific optimization level, `--lto-O2`.

Obviously, if more Kitsune-specific options were to be passed, more `-mllvm`
options would have to be provided. This is especially true with the GPU-centric
tapir targets that require the paths to
[libdevice bitcode files](glossary-libdevice-bitcode-file) to be provided
explicitly. However, the user need not concern themselves with these details
since Kitsune's driver handles it automatically [^3]. The code that does this
is in
{{'[`clang/lib/Driver/ToolChain.cpp`](https://{}/{}/kitsune/tree/{}/clang/lib/Driver/ToolChain.cpp)'.format(kitsune_repo_host, kitsune_repo_owner, kitsune_repo_branch)}}

This is, admittedly, a far from elegant design. However,
given the constraints under which Kitsune was developed, this was deemed
to be reasonable.

[^3]: In general, we do not expect most users to invoke the linker directly. In fact, even build systems such as [CMake](https://cmake.org) do not invoke the linker directly either - when linking object files during separate compilation, they, too, use the appropriate compiler driver.
