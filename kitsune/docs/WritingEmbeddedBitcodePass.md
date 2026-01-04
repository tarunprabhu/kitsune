# Writing an Embedded Bitcode Pass

[Embedded bitcode passes](glossary-embedded-bitcode) are LLVM
[passes](glossary-pass) that operate on
[embedded bitcode](glossary-embedded-bitcode). The design and implementation of
embedded bitcode as used by Kitsune is described in greater detail
[here](EmbeddedBitcode.md). Embedded bitcode passes are used extensively in
Kitsune's [pass pipeline](PassPipeline.md). This document describes how to
write such passes, both [in-tree](glossary-in-tree-pass) and
[out-of-tree](glossary-out-of-tree-pass) passes.

## Basic Structure

Embedded bitcode passes must inherit from the `EmbModulePass` class. Contrast
this with normal LLVM [module passes](glossary-module-pass) that inherit from
`PassInfoMixin`. Like `PassInfoMixin`, however, `EmbModulePass` also uses
the [CRTP](https://en.cppreference.com/w/cpp/language/crtp.html) idiom.

In addition, the pass class must implement a run method with the following
signature.

```c++
bool run(llvm::TTID tt,
         llvm::Module &m,
         llvm::ModuleAnalysisManager &am,
         llvm::Module &hostM,
         llvm::ModuleAnalysisManager &hostMAM);
```

The arguments that this method accepts are summarized in the table below.

```{table}
|||
|:-:|:-:|
| `tt` | The tapir target that generated the embedded module provided to the pass |
| `m` | The embedded module |
| `am` | Analysis manager for the embedded module, `m` |
| `hostM` | The host module in to which `m` is embedded |
| `hostMAM` | Analysis manager for the host module, `hostM` |
```

The method must return `true` if the pass modified the embedded module, `m`,
`false` if `m` was not modified.

```{important}
An embedded bitcode pass _**must not**_ modify the host module.
```

If the pass does not require any analyses, the alternative run method shown
below can be defined instead.

```c++
bool run(llvm::TTID tt,
         llvm::Module &m,
         llvm::Module &hostM,
         llvm::Module &hostMAM);
```

Note that this method does not accept an
[analysis manager](glossary-analysis-manager) for the embedded
module. However, an analysis manager for the host module is provided. Since the
embedded module analysis manager is not created, it may be marginally more
efficient to define this method when appropriate.

```{important}
Exactly **one** of the two `run` methods define above must be defined in the
pass.
```

Finally, the `run` method provided by the `EmbModulePass<T>` base class
*must* be explicitly used in the pass. This is the entry point that the new
pass manager will use when running the pass.

```c++
using EmbModulePass<EmbFuncNamesPass>::run;
```

Without this, the pass will not compile. The code below is a skeleton of an
embedded bitcode pass with the first entry point.

```c++
struct embFuncNamesPass : public EmbModulePass<EmbFuncNamesPass> {
  bool run(TTID tt, Module &m, ModuleAnalysisManager &am,
           Module &hostM, ModuleAnalysisManager &hostMAM) {
     ...
  }
  using EmbModulePass<EmbFuncNamesPass>::run;
};
```

## In-tree Pass

A complete example of a pass that prints the names of the functions in an
embedded module is shown below. Note that we use the second entry point
without an analysis manager for the embedded module.

```c++
#include <kitsune/Transforms/EmbModulePass.h>

using namespace llvm;

struct EmbFuncNamesPass : public EmbModulePass<EmbFuncNamesPass> {
  bool run(TTID tt, Module &m, Module &hostM, ModuleAnalysisManager &hostMAM) {
    for (Function &f : m.functions())
      outs() << f.getName() << "\n";
    return false;
  }

  using EmbModulePass<EmbFuncNamesPass>::run;
};
```

### Building

A `CMakeLists.txt` file that can be used to build this pass is shown below.
Here, we assume that the pass is part of a separate library,
`LLVMKitCustomPasses`. If, on the other hand, this were to be added to one of
the existing pass libraries, the source file need only be added to the list of
sources for the existing library.

```cmake
add_llvm_component_library(LLVMKitCustomPasses
  EmbFuncNamesPass.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  IRReader
  KitCore
  Passes
  Support
)
```

This will be built when Kitsune is built, so special configuration or build
commands are not required.


## Out-of-tree Pass

A complete example of a [pass plugin](glossary-pass-plugin) containing an
embedded bitcode pass is shown below. More information about writing and
building a pass plugin is provided [here](WritingPassPlugin.md).

```c++
#include <kitsune/Transforms/EmbModulePass.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

using namespace llvm;

struct EmbFuncNamesPass : public EmbModulePass<EmbFuncNamesPass> {
  bool run(TTID tt, Module &m, Module &hostM, ModuleAnalysisManager &hostMAM) {
    for (Function &f : m.functions())
      outs() << f.getName() << "\n";
    return false;
  }

  using EmbModulePass<EmbFuncNamesPass>::run;
};

template <typename Pass>
static void registerPass(ModulePassManager &pm, OptimizationLevel,
                         ThinOrFullLTOPhase) {
  pm.addPass(Pass());
}

static bool parsePassPipeline(StringRef name, ModulePassManager &pm,
                              ArrayRef<PassBuilder::PipelineElement>) {
  if (name == "print-emb-func-names") {
    pm.addPass(EmbFuncNamesPass());
    return true;
  }
  return false;
}

static void registerPasses(PassBuilder &pb) {
  pb.registerPostTapirEarlyCallback(registerPass<EmbFuncNamesPass>);
  pb.registerPipelineParsingCallback(parsePassPipeline);
}

extern "C" LLVM_ATTRIBUTE_WEAK ::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION,
    "EmbBitcodePassPluginDemo",
    "1.0",
    registerPasses};
}
```

### Building

A minimal `CMakeLists.txt` file to build the pass plugin is shown below. See
[here](WritingPassPlugin.md) for a discussion of this file, in particular the
compiler and linker options that have been provided and the packages that are
used.

```cmake
cmake_minimum_required(VERSION 3.20)
project(EmbFuncNamesPlugin LANGUAGES C CXX)

find_package(Kitsune CONFIG REQUIRED)

add_library(EmbFuncNamesPlugin SHARED
  EmbFuncNamesPlugin.cpp)

# Some preprocssor definitions are required when compiled file that include
# LLVM headers.
target_compile_definitions(EmbFuncNamesPlugin PRIVATE
  ${LLVM_DEFINITIONS})

# We need paths to the top-level include directories of both Kitsune and LLVM
# since the plugin requires headers from both.
target_include_directories(EmbFuncNamesPlugin PUBLIC
  ${KITSUNE_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS})

# If RTTI has been disabled in LLVM, it must be explicitly disabled when
# compiling the plugin. The code below will only work correctly when compiling
# the plugin with GCC or Clang.
if (NOT LLVM_ENABLE_RTTI)
  target_compile_options(EmbFuncNamesPlugin PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti>)
endif ()

# Since we have not linked any LLVM libraries, the plugin object will contain
# undefined symbols. On Darwin, this will raise an error unles we expicitly
# instruct the linker to allow undefined variables. On Linux and *BSD,
# undefined symbols are allowed by default.
target_link_options(EmbFuncNamesPlugin PUBLIC
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
```

The plugin can be configured and built as follows.

```shell
cmake -DCMAKE_C_COMPILER=/path/to/prefix/bin/kitcc \
      -DCMAKE_CXX_COMPILER=/path/to/prefix/bin/kit++ \
      -DCMAKE_PREFIX_PATH='/path/to/prefix/lib/cmake/kitsune;/path/to/prefix/lib/cmake/llvm'
      /path/to/dir/containing/CMakeLists.txt
```

```shell
make
```

### Usage

Assuming that the `CMakeLists.txt` file in the listing above was used to build
the plugin, it can be used in various ways. First, with one of Kitsune's
drivers.

```shell
kit++ --tapir=cuda \
      -fpass-plugin=/path/to/libEmbFuncNamesPlugin.so ...
```

It can also be used with `opt`. Unlike the drivers, `opt` does not run any
passes by default. To ensure that the passes in the plugin are run, they must be
specified explicitly.

```shell
opt --tapir=cuda \
    -passes='print-emb-func-names' \
    --load-pass-plugin=/path/to/libEmbFuncNamesPlugin.so ...
```

Since the pass has been registered with one of Kitsune's extension points, it
can also be run with the [kit-lowering](passes-kit-lowering) meta-pass

```shell
opt --tapir=cuda --tapir-cuda-arch=sm_80 \
    --tapir-cuda-runtime-bc=/path/to/cuda/libdevice.bc \
    -passes='kit-lowering<O2>' \
    --load-pass-plugin=/path/to/libEmbFuncNamesPlugin.so ...
```

Note that in all cases, we provided a [tapir target](glossary-tapir-target).
When running the `kit-lowering` meta-pass, additional options are required.
See [here](LLVMTools.md) for details on how LLVM's tools such as `opt` can be
used with Kitsune.

```{note}
In the examples above, the choice of the [cuda](tapir-targets-cuda) tapir target
was somewhat arbitrary. The [hip](tapir-targets-hip) tapir target would work
just as well, though the set of required command-line options would be
different. The tapir target **must** be one that uses embedded bitcode.
```
