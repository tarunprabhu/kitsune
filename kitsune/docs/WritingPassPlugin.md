# Writing a Pass Plugin

Pass plugins allow LLVM passes to be developed outside the LLVM source tree.
These are loaded from a dynamic shared object when Kitsune is used by Kitsune's
drivers, `kitcc`, `kit++` and `kitfc` and also LLVM's
[opt](https://llvm.org/docs/CommandGuide/opt.html) tool.
Pass plugins are only
supported on systems that allow dynamic loading of shared objects. Currently,
this includes all platforms that Kitsune supports.

This document focuses on what needs to be done to write a pass _plugin_. While
we provide some [background](writing-pass-plugin-background), and discuss some
aspects of the passes themselves, these are fairly limited. Consult LLVM's
[documentation](https://llvm.org/docs/WritingAnLLVMNewPMPass.html) for more
information.

(writing-pass-plugin-background)=
## Background

LLVM passes operate on the intermediate representations used by LLVM in the
middle and back-ends. They perform analyses, transformations and optimizations
that constitute some of the most important (and interesting) parts of the
compiler. A pass manager schedules the transformation passes and ensures that
the analysis passes used by the transformations are up-to-date.

There are two pass managers in use in LLVM. The older pass manager, referred to
as the "legacy pass manager" is used only in the LLVM backend. The passes
managed by this pass manager are known as legacy passes. The overwhelming
majority of passes that operate on LLVM-IR in the middle-end are managed by the
[new pass manager](https://blog.llvm.org/posts/2021-03-26-the-new-pass-manager/).
When we use the term "pass" in this document, we usually mean
passes managed by the new pass manager.

In this document, we only describe pass plugins consisting of pass that are
managed by the new pass manager.

LLVM supports a variety of pass kinds. Each of these operates at a different
level of granularity. These are summarized in the table below.

```{table}
| Kind | Description |
| :--: | :---------: |
| [Module Pass](https://llvm.org/docs/WritingAnLLVMPass.html#the-modulepass-class) | Passes that operate on an LLVM module as a whole and have no restrictions on the changes that they may make. |
| [Function Pass](https://llvm.org/docs/WritingAnLLVMPass.html#the-functionpass-class) | Passes that execute on each function in the module, independent of all other functions. They are only allowed to modify the functions on which they are operating. |
| [CallGraphSCC Pass](https://llvm.org/docs/WritingAnLLVMPass.html#the-callgraphsccpass-class) | Passes that traverse a module bottom-up on the call graph. |
| [Loop Pass](https://llvm.org/docs/WritingAnLLVMPass.html#the-looppass-class) | Passes that execute on each loop in a function independent of all other loops in the function. These are only allowed to modify the body of the loop on which they are executed. |
| [Region Pass](https://llvm.org/docs/WritingAnLLVMPass.html#the-regionpass-class) | Passes that execute on each single-entry-single-exit region in a function independent of all other such regions in the function. |
```

Pass plugins are likely to consist of either Function passes or Module passes.

```{note}
The links in the table above point to LLVM's documentation that was originally
written for the legacy pass manager. The classes, methods and functions
referenced there are not available in the new pass manager. Equivalent
functionality is available though. Some of these details are elaborated upon in
the documentation describing  for the new pass manager
[how to write a pass](https://llvm.org/docs/WritingAnLLVMNewPMPass.html)
for the new pass manager.
```

A few kinds of passes are not listed in the table above. One is an
[Immutable pass](https://llvm.org/docs/WritingAnLLVMPass.html#the-immutablepass-class)
that, like an analysis pass, does not change state. However it does not have to
be run and never needs to be updated.
Another is a
[MachineFunction Pass](https://llvm.org/docs/WritingAnLLVMPass.html#the-machinefunctionpass-class).
This executes on the machine-dependent representation of each LLVM function in a
module. Neither pass is discussed here since they are only supported by the
legacy pass manager.

## Basic Structure

The code below is a simple pass that runs on a module that prints the names of
all functions in an LLVM module.

```c++
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

using namespace llvm;

struct PrintFunctionsPass : PassInfoMixin<PrintFunctionsPass> {
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam) {
    for (Function& f : m)
      outs() << f.getName() << "\n";
    return PreservedAnalyses::all();
  }
};
```

Here, the `run` method is the entry point for the pass that will be invoked by
the pass manager. Depending on the kind of pass that you are writing, the
entry point will change. The table below lists the entry points for the various
kinds of passes

```{table}
| Kind | Description |
| :--: | :---------: |
| Module Pass | `PreservedAnalyses run(Module&, ModuleAnalysisManager&)` |
| Function Pass | `PreservedAnalyses run(Function&, FunctionAnalysisManager&)` |
| CallGraphSCC Pass |  `PreservedAnalyses run(LazyCallGraph::SCC&, CGSCCAnalysisManager&, LazyCallGraph&, CGSCCUpdateResult&)`|
| Loop Pass | `PreservedAnalyses run(Loop&, LoopAnalysisManager&, LoopStandardAnalysisResults&, LPMUpdater&)` |
| Region Pass | There only seem to be instances legacy region passes in LLVM's source. It is not clear if these passes are supported in the new pass manager  |
```

The returned value, `PreservedAnalyses::all()` indicates that the analyses
required by other transformation passes remain valid since this pass does not
modify the module in any way. To force all analyses to be recomputed, return
`PreservedAnalyses::none()`. One can selectively preserve individual analyses
or sets of analyses as well.

Pass plugins require a well-known entry point to be defined. This entry point
is called when the plugin is loaded by LLVM.

```c++
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

using namespace llvm;

struct PrintFunctionNamesPass : PassInfoMixin<PrintFunctionNamesPass> {
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam) {
    for (Function& f : m)
      outs() << f.getName() << "\n";
    return PreservedAnalyses::all();
  }
};

extern "C" LLVM_ATTRIBUTE_WEAK ::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION,
    "KitPassPluginDemo",
    "1.0",
    registerPasses};
}
```

```{important}
The name of the entry-point function and its signature must be _exactly_ as
shown. Otherwise, at [use-time](glossary-use-time), it may seem as if the
plugin is not being loaded. LLVM may not emit any diagnostic messages warning
of this.
```

We now describe the fields of the `llvm::PassPluginLibraryInfo` struct that is
returned by the entry point.

- The first field is an integer representing the API version required by the
  tapir target plugin. `LLVM_PLUGIN_API_VERSION` is the latest plugin API
  version supported by LLVM. We strongly recommend _always_ setting this to the
  latest version.

- The second field is the name of the plugin. This is mainly used when debugging
  the pass pipeline, but is not used for much else.

- The third field is a string and is the plugin version. This is only meaningful
  in the context of the plugin itself. We recommend adhering to the
  [semantic versioning specification](https://semver.org/).

- The fourth field is a callback that registers the pass with the pass manager.

The code below is an example of the `registerPasses` callback. The `PassBuilder`
class provides extension points at which dynamically loaded passes from pass
plugins
can be registered. The extension point should be chosen based on what the pass
does. For instance, if the pass perform peephole optimizations, the pass should
be registered with the "peephole" extension point. In the example below, the
`PrintFunctionNames` pass is registered with the "optimizer early" extension
point.

```c++
template <typename Pass>
static void registerPass(ModulePassManager &pm, OptimizationLevel) {
  pm.addPass(Pass());
}

static void registerPasses(PassBuilder &pb) {
  pb.registerOptimizerEarlyEPCallback(registerPass<PrintFunctionNamesPass>);
}
```

[LLVM's API reference](https://llvm.org/doxygen/classllvm_1_1PassBuilder.html)
for the `PassBuilder` class provides more information about the methods that can
be used to register passes with the various extension points. See the methods
whose names are of the form `register*Callback`. In addition to these, Kitsune
provides Kitsune-specific entry points for its
own pass pipelines. These are discussed in more detail [here](PassPipeline.md).

This approach only works when using the pass plugin with one of Kitsune's
drivers, that is `kitcc`, `kit++` and so on. Tools such as `opt` and `llc` also
support loading pass plugins. With these tools, we can explicitly specify
which passes to run. For this, an additional callback must be provided
that parses the value of the `-passes` option and determines which pass is to be
run. In the code below, we create an instance of `PrintFunctionNames` when
`print-function-names` is one of the passes provided to `-passes`.

```c++
static bool parsePassPipeline(StringRef name, ModulePassManager &pm,
                              ArrayRef<PassBuilder::PipelineElement>) {
  if (name == "print-function-names") {
    pm.addPass(PrintFunctionNamesPass());
    return true;
  }
  return false;
}

static void registerPasses(PassBuilder &pb) {
  pb.registerPipelineParsingCallback(parsePassPipeline);
}
```

In many case, it may be useful to register a pass with an extension point,
_and_ allow it to be run explicitly. To do so, simply call both `register`
methods as shown below.

```c++
static void registerPasses(PassBuilder &pb) {
  pb.registerOptimizerEarlyEPCallback(registerPass<PrintFunctionNamesPass>);
  pb.registerPipelineParsingCallback(parsePassPipeline);
}
```

The complete code for this simple pass plugin is shown below.

```c++
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

using namespace llvm;

struct PrintFunctionNamesPass : PassInfoMixin<PrintFunctionNamesPass> {
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam) {
    for (Function& f : m)
      outs() << f.getName() << "\n";
    return PreservedAnalyses::all();
  }
};

template <typename Pass>
static void registerPass(ModulePassManager &pm, OptimizationLevel,
                         ThinOrFullLTOPhase) {
  pm.addPass(Pass());
}

static bool parsePassPipeline(StringRef name, ModulePassManager &pm,
                              ArrayRef<PassBuilder::PipelineElement>) {
  if (name == "print-function-names") {
    pm.addPass(PrintFunctionNamesPass());
    return true;
  }
  return false;
}

static void registerPasses(PassBuilder &pb) {
  pb.registerOptimizerEarlyEPCallback(registerPass<PrintFunctionNamesPass>);
  pb.registerPipelineParsingCallback(parsePassPipeline);
}

extern "C" LLVM_ATTRIBUTE_WEAK ::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION,
    "PassPluginDemo",
    "1.0",
    registerPasses};
}
```

## Building with CMake

A minimal `CMakeLists.txt` file to build the pass plugin is shown below.

```cmake
cmake_minimum_required(VERSION 3.20)
project(PrintFunctionNamesPlugin LANGUAGES C CXX)

find_package(Kitsune CONFIG REQUIRED)

add_library(PrintFunctionNamesPlugin SHARED
  PrintFunctionNamesPlugin.cpp)

# Some preprocssor definitions are required when compiled file that include
# LLVM headers.
target_compile_definitions(PrintFunctionNamesPlugin PRIVATE
  ${LLVM_DEFINITIONS})

# We need paths to the top-level include directories of both Kitsune and LLVM
# since the plugin requires headers from both.
target_include_directories(PrintFunctionNamesPlugin PUBLIC
  ${KITSUNE_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS})

# If RTTI has been disabled in LLVM, it must be explicitly disabled when
# compiling the plugin. The code below will only work correctly when compiling
# the plugin with GCC or Clang.
if (NOT LLVM_ENABLE_RTTI)
  target_compile_options(PrintFunctionNamesPlugin PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti>)
endif ()

# Since we have not linked any LLVM libraries, the plugin object will contain
# undefined symbols. On Darwin, this will raise an error unles we expicitly
# instruct the linker to allow undefined variables. On Linux and *BSD,
# undefined symbols are allowed by default.
target_link_options(PrintFunctionNamesPlugin PUBLIC
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
```

Note that in the `CMakeLists.txt` file above, the LLVM libraries are not linked
into the shared library that is built. The plugin will be loaded by Kitsune's
compiler drivers, or LLVM's `opt` tool. In each case, the LLVM libraries will
have been linked into the loading tool. The undefined symbols in the plugin
shared object will be resolved at this time.

Unlike Linux and the BSD's, on MacOSX, the presence of undefined symbols will
result in a link-time error. The last two lines in the `CMakeLists.txt` file
above pass `-undefined dynamic_lookup` to the linker if the code is compiled
on MacOSX. This tells the linker that the undefined symbols will be resolved at
load-time.

If using a `CMakeLists.txt` file such as the one above, the plugin can be
configured as follows

```shell
cmake -DCMAKE_C_COMPILER=/path/to/prefix/bin/kitcc \
      -DCMAKE_CXX_COMPILER=/path/to/prefix/bin/kit++ \
      -DCMAKE_PREFIX_PATH='/path/to/prefix/lib/cmake/kitsune;/path/to/prefix/lib/cmake/llvm'
      /path/to/dir/containing/CMakeLists.txt
```

Here, `/path/to/prefix` is the path to the directory where Kitsune is
installed. If you have built Kitsune from source, you may also replace
`/path/to/prefix` with the path to the top-level build directory.

We have used `kit++` i.e. Kitsune's C++ frontend to compile the plugin.
This is **_not_** strictly necessary. You could also use `clang++` from the
Kitsune installation, or the C++ compiler that was used to build Kitsune itself.
Note that when determining the RTTI option to add (by examining
`LLVM_ENABLE_RTTI`), we assume that the compiler is either GCC or Clang.
Using a different compiler here may work, but this has not been tested.

In the configuration command above, we have also provided two paths to
`CMAKE_PREFIX_PATH`. These are paths to directories containing
`KitsuneConfig.cmake` and `LLVMConfig.cmake`. This is only necessary if these
paths are not in cmake's default search path.

## Building Manually

The `CMakeLists.txt` file provided above is only provided as an example. You
are not required to use [cmake](https://cmake.org) to build a pass plugin.
For single-file plugins, it may be convenient to just build it by hand as shown
below.

```shell
kit++ $(/path/to/prefix/bin/llvm-config --cppflags) \
      -fno-rtti \
      -shared -O1 -o PrintFunctionNamesPlugin.so PrintFunctionNamesPlugin.cpp
```

Here, the invocation of
[llvm-config](https://llvm.org/docs/CommandGuide/llvm-config.html)
will return the preprocessor options
required to compile code that includes LLVM headers. This includes the path to
the top-level include directory. You may also consider using the `--cxxflags`
option which will also set the minimum C++ version required by LLVM.

We also assume that RTTI has been disabled in the Kitsune compiler that has been
built. It would be better to use `llvm-config --has-rtti` here to check if
`-fno-rtti` should be added.

```{tip}
In our experience, it is nearly always better to use a build system such as
[CMake](https://cmake.org) or [Meson](https://mesonbuild.com), or even a simple
[MakeFile](https://www.gnu.org/software/make/manual/make.html) instead of
compilng manually.
```

## Usage

The plugin can be used with a Kitsune driver as shown below.

```shell
kit++ -fpass-plugin=/path/to/plugin.so ...
```

The plugin can also be used with `opt`. Unlike the drivers, `opt` does not run
any passes by default. To ensure that the passes in the plugin are run, they
must be specified explicitly.

```shell
opt -passes='print-function-names' --load-pass-plugin=/path/to/plugin.so ...
```

If the pass has been registered with one of the other extension points, for
instance "optimizer early" as shown in the example above, one of the standard
optimization levels may also be used.

```shell
opt -O0 --load-pass-plugin=/path/to/plugin.so ...
```
