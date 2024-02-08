# Writing a Tapir Target Plugin #

Tapir target plugins allow custom tapir targets to be written outside Kitsune
and loaded from a dynamic shared object (DSO). In principle, these are similar
to LLVM's [pass plugins](WritingPassPlugin.md). A sample tapir target plugin is
provided in the
{{'[examples directory](https://{}/{}/kitsune/tree/{}/kitsune/examples/TTPluginDemo)'.format(kitsune_repo_host, kitsune_repo_owner, kitsune_repo_branch)}}.
This document provides more information about writing and building such a
plugin.

```{note}
Tapir target plugins are only available on systems that support dynamically
loaded shared objects. Currently, this includes all systems that Kitsune
supports.
```

## Basic Structure

The core functionality of the custom tapir target is contained within a class
that inherits from the
{{'[`TapirTarget`](https://{}/{}/kitsune/tree/{}/llvm/include/llvm/Transforms/Tapir/Loweringutils.h)'.format(kitsune_repo_host, kitsune_repo_owner, kitsune_repo_branch)}}
base class. The skeleton of such a custom tapir target is shown in the code
block below.

```c++
#include <kitsune/Core/TTPlugin.h>
#include <llvm/Transforms/Tapir/LoweringUtils.h>

using namespace llvm;

class CustomTT : public TapirTarget {
public:
  Custom(Module &m, const TTOptions &tto);
  virtual ~CustomTT() = default;

  /* Override virtual functions */
};
```

We do not describe all the virtual functions defined in `TapirTarget` that
may - or must - be overridden  in the derived class. For details on
each of these callbacks, see the inline documentation associated with the
`TapirTarget` class [^1].

A custom tapir target may require a custom loop outline processor.
This is a class that inherits from the
{{'[`LoopOutlineProcessor`](https://{}/{}/kitsune/tree/{}/llvm/include/llvm/Transforms/Tapir/Loweringutils.h)'.format(kitsune_repo_host, kitsune_repo_owner, kitsune_repo_branch)}}
class. In this case, the `getLoopOutlineProcessor()` method of the tapir
target must be overridden as shown.

```c++
#include <kitsune/Core/TTPlugin.h>
#include <llvm/Transforms/Tapir/LoweringUtils.h>

using namespace llvm;

class CustomLOP : public LoopOutlineProcessor {
  CustomLOP(Module &m, const TTOptions &tto);
  ~CustomLOP();

  /* Override virtual functions */
};

class CustomTT : public TapirTarget {
public:
  Custom(Module &m, const TTOptions &tto);
  virtual ~CustomTT() = default;

  /* Override virtual functions */

  LoopOutlineProcessor *
  getLoopOutlineProcessor(const TapirLoopInfo *tl) override final {
    return new BookendLOP(M, this->getOptions());
  }
};
```

A custom loop outline processor is _**not**_ required, If one has not been
provided, that is, if the `getLoopOutlineProcessor()` method is not overridden,
Kitsune will automatically use a default loop outline processor.

Now that the custom tapir target has been defined, we define the well-known
entry point function that must be provided in a tapir target plugin.

```c++
static TapirTarget* getTapirTarget(Module &hostM, const TTOptions &tto) {
  return new CustomTT(hostM, tto);
}

static TTPlugin::ExtraArgsList getCompilerOptions() {
  return { "-O" };
}

static TTPlugin::ExtraArgsList getLinkerOptions() {
  return { };
}

extern "C" ::llvm::TTPluginInfo LLVM_ATTRIBUTE_WEAK llvmGetTTPluginInfo() {
  return {
    /* PluginAPIVersion= */ LLVM_TTPLUGIN_API_VERSION,
    /* PluginName= */       "TTPluginDemo",
    /* PluginVersion= */    "1.0",
    /* makeTapirTarget= */  getTapirTarget,
    /* getCompilerOpts= */  getCompilerOptions,
    /* getLinkerOpts= */    getLinkerOptions,
  };
}
```

```{important}
The name of the entry-point function and its signature must be _exactly_ as
shown. Otherwise, at [use-time](glossary-use-time), it may seem as if the
plugin is not being loaded. LLVM may not emit any diagnostic messages warning
of this.
```

We now describe the fields of the `llvm::TTPluginInfo` struct that is returned
by the entry point.

- The first field is an integer representing the API version required by the
  tapir target plugin. Currently, Kitsune only supports a single API version for
  tapir target plugins. This is `LLVM_TTPLUGIN_API_VERSION` which is defined to
  be 1.

- The second field is the name of the plugin. Kitsune's does not examine this
  field. It is, therefore, primarily useful for debugging. To reduce the risk
  of issues should Kitsune examine this name in the future, avoid using spaces
  in the name.

- The third field is a string and is the plugin version. This is only meaningful
  in the context of the plugin itself. Kitsune's does not, currently, examine
  the version. To reduce the risk of issues should Kitsune example this version
  in the future, we recommend adhering to the
  [semantic versioning specification](https://semver.org/).

- The fourth field is a callback that creates and returns an instance of the
  custom tapir target class. The caller of this callback will take ownership
  of the returned object.

- The fifth field is a callback that returns a vector of strings. Each element
  of this vector must be a command-line option that is known to the
  language-specific compiler - such as `-cc1` or `-fc1` - with which this tapir
  target plugin will be used. The callback will be invoked by Kitsune's driver.
  For details, see the section on [using a tapir target plugin](#usage). In the
  example here, the `-O` option will be added to the invocation of the
  compiler.

  ```{note}
  When using a tapir target plugin directly with
  [opt](https://llvm.org/docs/CommandGuide/opt.html), this callback will not
  be used.
  ```

- The sixth field is a callback that returns a vector of strings. This callback
  will be invoked by Kitsune's driver. The options in the returned vector will
  be added to the invocation of the linker.
  For details, see the section on [using a tapir target plugin](#usage).
  In the example here, an empty vector is returned indicating that the plugin
  does not inject any additional options to the linker command line.

  ```{note}
  When using a tapir target plugin directly with `opt`, this callback will not
  be used.
  ```

  ```{tip}
  Since the default linker that is used will vary across platforms, if this
  callback is expected to return a non-empty list, it may be necessary to use
  LLD as the linker to ensure that the returned options will be valid across
  platforms. LLD is guaranteed to be available when Kitsune is built since
  Kitsune requires it. In this case, `-fuse-ld=lld` must be specified whenever
  this tapir target plugin is used.
  ```

The code below is a complete listing of all the code that has been shown thus
far.

```c++
#include <kitsune/Core/TTPlugin.h>
#include <llvm/Transforms/Tapir/LoweringUtils.h>

using namespace llvm;

class CustomLOP : public LoopOutlineProcessor {
public:
  CustomLOP(Module &m, const TTOptions &tto);
  ~CustomLOP();

  /* Override virtual functions */
};

class CustomTT : public TapirTarget {
public:
  CustomTT(Module &m, const TTOptions &tto);
  virtual ~CustomTT() = default;

  /* Override virtual functions */

  LoopOutlineProcessor *
  getLoopOutlineProcessor(const TapirLoopInfo *tl) override final {
    return new CustomLOP(M, this->getOptions());
  }
};

static TapirTarget* getTapirTarget(Module &hostM, const TTOptions &tto) {
  return new CustomTT(hostM, tto);
}

static TTPlugin::ExtraArgsList getCompilerOptions() {
  return { "-O" };
}

static TTPlugin::ExtraArgsList getLinkerOptions() {
  return { };
}

extern "C" ::llvm::TTPluginInfo LLVM_ATTRIBUTE_WEAK llvmGetTTPluginInfo() {
  return {
    /* PluginAPIVersion= */ LLVM_TTPLUGIN_API_VERSION,
    /* PluginName= */       "TTPluginDemo",
    /* PluginVersion= */    "1.0",
    /* makeTapirTarget= */  getTapirTarget,
    /* getCompilerOpts= */  getCompilerOptions,
    /* getLinkerOpts= */    getLinkerOptions,
  };
}
```

## Building with CMake

A minimal `CMakeLists.txt` file to build a tapir target plugin is shown below.

```cmake
cmake_minimum_required(VERSION 3.20)
project(CustomTT LANGUAGES C CXX)

find_package(Kitsune CONFIG REQUIRED)

add_library(CustomTT SHARED
  CustomTT.cpp)

# Some preprocssor definitions are required when compiled file that include
# LLVM headers.
target_compile_definitions(CustomTT PRIVATE
  ${LLVM_DEFINITIONS})

# We need paths to the top-level include directories of both Kitsune and LLVM
# since the plugin requires headers from both.
target_include_directories(CustomTT PUBLIC
  ${KITSUNE_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS})

# If RTTI has been disabled in LLVM, it must be explicitly disabled when
# compiling the plugin. The code below will only work correctly when compiling
# the plugin with GCC or Clang.
if (NOT LLVM_ENABLE_RTTI)
  target_compile_options(CustomTT PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti>)
endif ()

# Since we have not linked any LLVM libraries, the plugin object will contain
# undefined symbols. On Darwin, this will raise an error unless we explicitly
# instruct the linker to allow undefined variables. On Linux and *BSD,
# undefined symbols are allowed by default.
target_link_options(CustomTT PUBLIC
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
are not required to use [cmake](https://cmake.org) to build a tapir target
plugin. For single-file plugins, it may be convenient to just build it by hand
as shown below.

```shell
kit++ $(/path/to/prefix/bin/llvm-config --cppflags) \
      -fno-rtti \
      -shared -O1 -o CustomTT.so CustomTT.cpp
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

The plugin can be used with a Kitsune driver and also with LLVM's `opt` tool.

```shell
kit++ --tapir=custom --tapir-plugin=/path/to/plugin.so ...
```

Note that the plugin can **_only_** be used with the
[custom](tapir-targets-custom) tapir target. Attempting to use it with any
other tapir target - or failing to provide the `--tapir=` option will result in
an error.

```shell
opt --tapir=custom --tapir-plugin=/path/to/plugin.so ...
```

Note that even when using `opt`, the command-line options do not change. This is
not the case with other LLVM plugins, in particular, the
[pass plugins](WritingPassPlugin.md).

In the examples above, any other constraints imposed by Kitsune have been
omitted for clarity. For instance, Kitsune's compiler drivers may require
optimizations to be  enabled when using the `custom` tapir target. In this case,
at least `-O1` may have to be provided.

[^1]: The `TapirTarget` is a core component of the tapir parallel IR extensions to LLVM-IR and is currently not documented here. We might do so at some point.
