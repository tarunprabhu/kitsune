# Getting Started

This document describes how to obtain and build Kitsune. The only way to obtain
Kitsune is to build it from source. We do not provide pre-built binaries.
Since Kitsune is built using LLVM, much of LLVM's
[documentation](https://llvm.org/docs) will be relevant for Kitsune, and we will
make frequent references to it here rather than repeating the material.
Divergences from LLVM will be stated explicitly.

## **Requirements**

Compiling Kitsune is very demanding of the host system - both the software
and the hardware. Before beginning, please review the requirements below.

### **Hardware**

Kitsune has only been tested on the following platforms. It may work on
platforms not listed here, but that will almost certainly require some
modifications to the build system. Compilers other than those listed below may
be capable of compiling Kitsune, but they have not been tested.

```{table}
| OS    | Architecture | Compilers |
| :---: | :----------: | :-------: |
| FreeBSD | amd64 | GCC, Clang |
| Linux | amd64 | GCC, Clang |
| Linux | aarch64 | GCC, Clang |
| MacOSX | arm64 | Clang |
```

There are no plans to support Kitsune on Windows.

### **Software**

Kitsune requires several software packages to be installed. These are
enumerated [here](https://llvm.org/docs/GettingStarted.html#software). In
addition to the packages listed there, Kitsune may need additional packages
depending on the tapir targets that are enabled and other configuration
options. These will be listed in the tapir-target-specific sections below.
At a minimum the following may be required:

- **patch** --- apply a diff file to an original
- **wc** --- print newline, word, and byte counts for files

On Unix-like systems, these should be present by default on most installations.
If they are not, they should be readily available in the system's package
manager.

(getting-started-host-cxx-toolchain)=
#### **Host C++ Toolchain**

Kitsune has only been built with fairly modern C++ compilers. The oldest
versions that are known to work are GCC 13.1 and Clang 16. Older versions of
these compilers may also work but have not been tested. LLVM's
[documentation](https://llvm.org/docs/GettingStarted.html#host-c-toolchain-both-compiler-and-standard-library)
provides more detailed information on how to obtain a C++ toolchain suitable
for building LLVM.

### **Tapir target dependencies**

Some tapir targets require dependencies that must be available before Kitsune
can be built. These are only required if the corresponding tapir targets are
enabled. In some cases, Kitsune will automatically fetch and build the
dependencies for a tapir target as part of the build process. Otherwise, the
user must ensure that a suitable installation is available. Failure to do so
will result in a configure-time or build-time error. This is summarized in the
table below.

```{table}
|Tapir<br>Target| Requires | Versions | Manual Installation Required|
|:--------------| :------: | :------: | :--------------------: |
| [cuda](tapir-targets-cuda) | [CUDA](https://developer.nvidia.com/cuda-downloads) | {{kitsune_cuda_version_min}} - {{kitsune_cuda_version_max}} | **Yes** ([details](#cuda)) |
| [custom](#tapir-targets-custom) | N/A | N/A | Plugin dependencies are not required when building Kitsune but must be available when compiling user code with this tapir target |
| [hip](tapir-targets-hip) | [ROCm](https://rocm.docs.amd.com/en/latest) | {{kitsune_hip_version_min}} - {{kitsune_hip_version_max}} | **Yes** ([details](#hip)) |
| [nolo](tapir-targets-nolo) | - | - | - |
| [opencilk](tapir-targets-opencilk) | [Cheetah](https://github.com/OpenCilk/cheetah) | N/A | **No** (fetched and built automatically when building Kitsune) |
| [pthreads](tapir-targets-pthreads) | pthreads | any | **No** (available by default on supported platforms) |
| [serial](tapir-targets-serial) | - | - | - |
```

The dependencies for certain tapir targets may not be available on all systems.
See [this table](tapir-targets-table-platforms) for a summary of the systems and
architectures on which the tapir targets may be enabled.

#### cuda

The cuda tapir target requires the
[NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit).
On Linux, the easiest way to obtain this is via the
package manager of your distribution. The name of the package, if available,
varies
depending on the distribution. The table below lists the names of the required
package on some distributions.

```{table}
| Distribution | Package |
| :----------- | :------ |
| Arch | cuda |
| Debian | nvidia-cuda-toolkit |
| Gentoo | nvidia-cuda-toolkit |
```

For definitive instructions on how to install NVIDIA's CUDA toolkit, consult
your distribution's documentation. Note that only a fairly narrow range of
cuda toolkit versions are supported. If the package provided by
your distribution does not fall in this range, you may have to download and
install it [manually](https://developer.nvidia.com/cuda-downloads).

#### hip

The hip tapir target requires AMD's
[ROCm](https://rocm.docs.amd.com/en/latest/). On Linux, easiest way to obtain
this is via the package manager of your distribution. The name of the package,
if available, varies depending on the distribution. The table below lists the
names of the required packages on some distributions.

```{table}
| Distribution | Package |
| :----------- | :------ |
| Arch | rocm-core |
| Debian | rocm |
| Gentoo | rocm-core |
```

For definitive instructions on how to install AMD's ROCm, consult your
distribution's documentation. Note that only a fairly narrow range of ROCm
versions are supported. If the package provided by your distribution does not
fall in this range, you may have to download and install it
[manually](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).

## **Obtaining Kitsune**

Kitsune's source code must be checked out using [git](https://git-scm.com/). The
most straightforward to do so is:

{{command_clone_html}}

Alternatively, the repo can be cloned using SSH.

{{command_clone_ssh}}

This will checkout the entire repo. You may need to switch to the current
branch.

{{command_checkout}}

A full checkout of Kitsune can be very large - at the time of writing, this can
be larger than 7GB. A
[shallow clone](https://git-scm.com/docs/git-clone#Documentation/git-clone.txt---depthltdepthgt)
may be used to save disk space and reduce clone times. The command below
will clone only the current branch.

{{command_clone_current}}

Alternatively, only the latest commit in the current branch may be cloned

{{command_clone_current_head}}

In either of these cases, the `git checkout` command is not required.

```{note}
If you are interested in contributing to Kitsune, a full checkout using SSH
is recommended.
```

## **Building Kitsune**

Only cmake's [Ninja](https://cmake.org/cmake/help/latest/generator/Ninja.html)
and
[Unix Makefiles](https://cmake.org/cmake/help/latest/generator/Unix%20Makefiles.html)
generators have been tested, though others might also work. If using the former,
[ninja](https://ninja-build.org/) must be available on the system, while the
latter requires a suitable implementation of the
[`make`](https://en.wikipedia.org/wiki/Make_(software)#Variants), such as
[GNU Make](https://www.gnu.org/software/make/). In the remainder of this
section, we will assume that the Ninja generator is used.

At a bare minimum, building Kitsune involves [configuring](#configure),
then [building](#build) and, optionally, [installing](#install). The commands
below show roughly what this might entail. More details follow in the sections
for each step.

```
mkdir build-directory
cd build-directory
cmake -G Ninja /path/to/kitsune/checkout/llvm
ninja
ninja install
```

### **Configure**

Kitsune requires that it be built outside of the source directory. In other
words, cmake must *not* be run in the same directory in which Kitsune was
checked out. While building Kitsune in a subdirectory of the source is
permitted, we recommend building Kitsune in a separate directory altogether.

Assuming that Kitsune has been checked out in `/path/to/kitsune`, one could
setup a build directory in `/path/to/kitsune-build`

```
mkdir /path/to/kitsune-build
cd /path/to/kitsune-build
```

The general form of the configure command is as follows

```
cmake -G Ninja [OPTIONS] /path/to/kitsune/llvm
```

Here, `[OPTIONS]` are the desired configuration options. These will be
discussed presently. Meanwhile, note that the last argument on the command
line above is `/path/to/kitsune/llvm`. The trailing `llvm` is required. Without
it, cmake will raise an error.

A basic configuration command that one could use is shown here

```
cmake -G Ninja \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/where/to/install/kitsune \
      -DKITSUNE_ENABLE_LANGS=<semicolon-separated-list> \
      -DKITSUNE_ENABLE_TAPIR_TARGETS=<semicolon-separated-list> \
      /path/to/kitsune/llvm
```

Note that we have used `clang` and `clang++` that are assumed to be in `$PATH`
here. On systems where they are available, `gcc` and `g++` may also be used.

A number of options are available to customize Kitsune's configuration. In most
cases, you will need to explicitly provide one or more of these. In addition,
most of
[LLVM's configuration options](https://llvm.org/docs/CMake.html#frequently-used-llvm-related-variables) may also be used,
[albeit with some caveats](#modified-llvm-cmake-options). A list of Kitsune-specific
configuration options follow. The values in parentheses are the default values
for these options. We begin with the options that are likely to be used by
most users. Following the default value in parentheses is the cmake type of
the option.

- **KITSUNE_ENABLE_TAPIR_TARGETS** ("{{kitsune_default_tapir_targets}}") : `STRING`

    A semicolon-separated list of tapir targets that should be built. The
    following list contains the elements that may be added to this list.

    {{'```[{}]```'.format(kitsune_default_tapir_targets_list)}}

    By default, all tapir targets in this list are built. These are the
    "non-universal" tapir targets. The "universal" tapir targets are always
    built. The complete list of supported tapir targets can be found
    [here](tapir-targets-table-platforms). In order to build only the universal
    tapir targets, the following may be passed to cmake

    ```
    -DKITSUNE_ENABLE_TAPIR_TARGETS=""
    ```

- **KITSUNE_ENABLE_LANGS** ("{{kitsune_default_langs}}") : `STRING`

    A semicolon-separated list of languages for which Kitsune frontends should
    be built. The following table lists the supported elements of this list.

    ```{table}
    |    | Language | Comments |
    |:-: | :------: | :------: |
    | c | C | This is mandatory |
    | cxx | C++ | This is mandatory |
    | fortran | Fortran | |
    ```

    By default, only the C and C++ frontends are built. These are also
    mandatory. In other words, if any other languages are added to the list,
    'c' and 'cxx' _must_ be present.

- **KITSUNE_CUDA_PREFIX** (`""`) : `STRING`

    The [cuda](tapir-targets-cuda) tapir target requires a supported version of
    [NVIDIA's CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). On some
    systems, if the toolkit is installed from the system's package manager,
    the `CUDA_PATH` environment variable will have been set. In this case,
    Kitsune will be able to automatically detect the cuda installation and will
    use it. However, if either the `CUDA_PATH` environment variable is not set,
    or if it points to an unsupported installation, the `-DKITSUNE_CUDA_PATH`
    option must be provided. The value must be the path to the root of a
    suitable installation of the CUDA Toolkit.

- **KITSUNE_HIP_PREFIX** (`""`) : `STRING`

    The [hip](tapir-targets-hip) tapir target requires a supported version of
    [AMD's ROCm](https://rocm.docs.amd.com/en/latest/). On some systems, if
    this has been installed from the system's package manager,
    the `ROCM_PATH` environment variable will have been set. In this case,
    Kitsune will be able to automatically detect the cuda installation and will
    use it. However, if either the `ROCM_PATH` environment variable is not set,
    or if it points to an unsupported installation, the `-DKITSUNE_HIP_PATH`
    option must be provided. The value must be the path to the root of a
    suitable installation of ROCm.

- **KITSUNE_BUILD_DOCS** (`OFF`) : `BOOL`

    Set this to `ON` to build Kitsune's documentation. This requires a number
    of packages to be installed including
    [sphinx](https://www.sphinx-doc.org/en/master/),
    [doxygen](https://www.doxygen.nl/),
    [myst-parser](https://myst-parser.readthedocs.io/en/latest/), among others.
    A complete list can be found
    [here](https://github.com/llvm/llvm-project/tree/main/llvm/docs/requirements.txt).

- **KITSUNE_GCC_INSTALL_DIR** (`""`) : `STRING`

    On Linux, and some other platforms, Kitsune requires a GCC installation from
    which it uses object files
    containing code to bootstrap program execution (`crt0.o`, `crt1.o` etc.),
    the C++ standard library (`libstdc++`), the start of constructors and
    destructors (`crtbegin.o`, `crtend.o`) etc. The oldest version of GCC that
    Kitsune is known to work with
    is GCC 13.1. It may work with older versions, but that has not been tested.
    If the standard GCC installation on a system is older than the minimum
    supported version, this option must be provided with the value set to the
    path of a directory
    containing `libgcc.a`. If GCC was installed to `/gcc_install`, `libgcc.a`
    will often be found in
    `/gcc_install/lib/gcc/<triple>/<version>` where `<triple>` is the target
    triple for the system and `<version>` is the GCC version. If a suitable
    version of GCC is not found on the system, and this variable is not set to
    a valid path, Kitsune may fail to build, even if configuring it succeeds.

- **KITSUNE_SYSROOT** (`""`) : `STRING`

    Kitsune requires a fairly modern "sysroot" i.e. an installation of the
    C standard library (both headers and object files). In cases where the
    C standard library provided by the system is very old (such as on many
    HPC systems) or not present at all (such as MacOSX), this option must be
    provided with the path to the root of an installation of the C standard
    library and binutils. On MacOSX, passing the following when configuring
    Kitsune is often sufficient

    ```
    -DKITSUNE_SYSROOT=$(xcrun --show-sdk-path)
    ```

    Without this, Kitsune may fail to build. Even if the build succeeds,
    [running the tests](#check) may fail. If building Kitsune (or running
    Kitsune's tests) fails with a message similar to the following

    ```
    version `GLIBCXX_<N>.<M>.<P>' not found
    ```

    where `<N>`, `<M>`, and `<P>` are some numbers, it is likely that you
    need to set `KITSUNE_SYSROOT` to an appropriate path containing a more
    [modern toolchain](getting-started-host-cxx-toolchain)


In addition to these, some other configuration options are available that may
be useful in some cases, but most users should not have to modify the default
values of these.

- **KITSUNE_BUILD_EXAMPLES** (`ON`) : `BOOL`

    The code in the `kitsune/examples` directoryincludes pass plugins, tapir
    target plugins and other
    code that may be useful in understanding how to use Kitsune. However,
    building them typically requires a fairly robust, modern toolchain
    (including system headers and a fairly modern binutils) to be available
    when building Kitsune. On systems where these are not available when
    building Kitsune, building these may fail. In such cases,
    setting `-DKITSUNE_BUILD_EXAMPLES=OFF` will skip the `kitsune/examples`
    directory altogether.

    ```{warning}
    If the examples are skipped, some tests will not be run. This is not ideal
    since any issues with the installation will not be exposed early. As a
    result, we do not recommend setting `-DKITSUNE_BUILD_EXAMPLES=OFF` unless
    absolutely necessary
    ```

- **KITSUNE_INCLUDE_TESTS** (`ON`) : `BOOL`

    If `-DKITSUNE_INCLUDE_TESTS=OFF` is provided, Kitsune's tests will not be
    built. This can speedup the process of building Kitsune slightly.

- **KITSUNE_ENABLE_EXPERIMENTS** (`ON`) : `BOOL`

    By default, Kitsune will descend into the `kitsune/experiments` directory.
    At this time, this results in a file being created in the source directory
    that can be used to manually build and run the examples in that directory.
    These are _not_ used by Kitsune's standard tests, so if a pristine source
    directory is required, `-DKITSUNE_ENABLE_EXPERIMENTS=OFF` may be used. The
    code in the `kitsune/experiments` can still be built by hand in this case,
    but the  associated Makefile-based scripts will not work.


#### Recommended LLVM CMake Options ####

Some cmake options from LLVM are recommended, but certainly not required. These
are listed below.

- **LLVM_CCACHE_BUILD** (`OFF`): `BOOL`

    We recommend a compiler cache when building Kitsune. LLVM's build system can
    automatically handle [ccache](https://ccache.dev/) if the following option
    is passed to cmake when configuring Kitsune

    ```
    -DLLVM_CCACHE_BUILD=ON
    ```

    This can speedup recompilations of Kitsune and is _strongly_ recommended if
    you are interested in modifying Kitsune. If you only need to build Kitsune,
    it may still be worth enabling `ccache` if you intend to periodically pull
    from Kitsune's repo.

    ```{note}
    Note that `ccache` may require 200-500MB of disk space
    ```

    Other compiler caches such as [sccache](https://github.com/mozilla/sccache)
    may also work. However, none of these have been tested. They may also have
    to be explicitly enabled by setting `-DCMAKE_C_COMPILER_LAUNCHER` and
    `-DCMAKE_CXX_COMPILER_LAUNCHER`.

- **LLVM_ENABLE_ASSERTIONS** (`OFF`): `BOOL`

    Assertions in LLVM's source code are automatically enabled in
    [Debug](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)
    builds
    and disabled in
    [Release](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)
    builds. For Kitsune
    developers, we _strongly_ recommend enabling assertions explicitly on both
    Debug and Release builds.

    ```
    -DLLVM_ENABLE_ASSERTIONS=ON
    ```

    If assertions are enabled, a programmer error that would otherwise result in
    a compiler crash will often trigger an associated assertion. These will
    often have an associated diagnostic message that can help narrow down the
    root cause of the error.

- **LLVM_LINK_LLVM_DYLIB** (`OFF`): `BOOL`

    Linking LLVM's tools against `libLLVM` instead of statically linking against
    individual static archives (libLLVMSupport.a, libLLVMBitWriter.a and so on)
    can reduce link times. This is typically more useful for Kitsune's developers

    ```
    -DLLVM_LINK_LLVM_DYLIB=ON
    ```

- **LLVM_TARGETS_TO_BUILD** (`"all"`): `STRING`

    This cmake option controls which backends to build. By default, all backends
    known to LLVM are built. For users and developers who are not interested in
    cross-compiling, setting

    ```
    -DLLVM_TARGETS_TO_BUILD=host
    ```

    will only build the backend for the machine on which Kitsune is being built.
    This can substantially decrease build times and reduce the amount of disk
    space used by the built object files.

    ```{warning}
    When developing for Kitsune, setting the `-DLLVM_TARGETS_TO_BUILD=host` can
    sometimes hide bugs inadvertently introduced into other backends by
    modifications made earlier in the compilation pipeline.
    ```

    Some tapir targets require additional backends. The table below summarized
    the backends required by such tapir targets.

    ```{table}
    | Tapir Target | Backend |
    | :----------: | :-----: |
    | [cuda](tapir-targets-cuda) | NVPTX |
    | [hip](tapir-targets-hip) | AMDGPU |
    ```

    Even if these targets have been enabled, the corresponding backends do not
    have to be added to `LLVM_TARGETS_TO_BUILD`. Kitsune's build system will
    automatically add the required backends. See [here](BuildSystem.md) for more
    details about Kitsune's build system.


#### Modified LLVM CMake Options ###

If you have built LLVM in the past, note that there are some other differences
in the way Kitsune is built and in how some of LLVM's configuration options are
handled. In general, most of
[LLVM's configuration options](https://llvm.org/docs/CMake.html#frequently-used-llvm-related-variables)
are supported. A few, such as `BUILD_SHARED_LIBS` are not supported - setting
them to a non-default value will result in a configure-time error. Others,
such as `LLVM_BUILD_LLVM_DYLIB` have different defaults in Kitsune. A complete
list of unsupported and modified options are
[provided](getting-started-modified-llvm-cmake-options).


(getting-started-modified-llvm-cmake-options)=
```{table}
| Option | Default (Kitsune) | Default (LLVM) | Comments |
| :----- | :---------------- | :------------- | :------- |
| `BUILD_SHARED_LIBS` | `OFF` | `OFF` | Settings this to `ON` will result in a configure-time error |
| `LLVM_BUILD_LLVM_DYLIB` | `ON` | `OFF` | The user-provided value of this parameter is effectively ignored. Kitsune's build system will force this to be set to `ON` |
| `LLVM_ENABLE_PROJECTS` | `"clang;kitsune;lld"` | `""` | Projects strictly required by Kitsune will always be built. Optional (from Kitsune's perspective) projects may be provided by the user. These will be respected
```

For more information about the design and behavior of Kitsune's build system,
see [this document](BuildSystem.md).


### **Build**

If the Ninja generator was used when configuring Kitsune, simply running

```
ninja
```

is sufficient to build Kitsune. There are some caveats to be aware of if
Fortran support has been enabled. This will be discussed
[later](#caveats-when-building-kitsune-with-fortran-enabled).

If the 'Unix Makefiles' generator was used when configuring Kitsune (this is
the default if the `-G` option was not provided), the
`make` command should be used.

Note that, unlike `ninja`, parallel builds may _not_ be the default in all
`make` implementations. Building in parallel must be requested explicitly. In
the case of GNU make, the following command is the equivalent of invoking
`ninja` without any other command-line options

```
make -j
```

Some `make` implementations, notably on FreeBSD require an explicit number of
parallel jobs to be provided to the `-j` option.

```
make -j 8
```

#### Caveats when building Kitsune with Fortran enabled

Enabling Fortran support drastically increases the amount of memory required
to build Kitsune since this requires building [flang](https://flang.llvm.org).
Flang requires [MLIR](https://mlir.llvm.org). Both these projects, and `flang`
in particular require a lot of memory to compile. For reference, on a machine
with 16 cores and 32GB of RAM running Linux, simply invoking `ninja` is
sufficient to easily build `clang`, `llvm` and `lld` (this will spawn as many
parallel jobs as there are visible CPU's, so 16 in this case). On the other
hand, when compiling both MLIR and, even more so, `flang`, spawning 16 parallel
jobs with 32GB of RAM results in the system running out of memory
and swapping to disk (if a swap partition is active). In general, to avoid
swapping, we recommend

```
ninja -j 6
```

When building on a machine with more RAM, this number can be increased. A rule
of thumb is to assume that each compile job in `flang` will require 3GB of RAM,
with link jobs requiring even more. Building MLIR does not put as much pressure
on memory as `flang`, though building it requires substantially more memory
than building `clang` or `llvm`.


### **Check**

Running Kitsune's checks after a successful build is not required, but it is
recommended.

```
ninja check-kitsune-all
```

This will run the Kitsune-specific tests as well as on the subprojects required
by Kitsune. The command below shows the individual `check-` commands for each
project. These may be run separately if one wishes.

```
ninja check-kitsune check-clang check-llvm check-lld
```

If Fortran support has been enabled, the tests of MLIR and Flang are also run.
These can also be run explicitly.

```
ninja check-mlir check-flang
```

Unlike the builds, the tests of MLIR and Flang are not as
memory-intensive, so running them without explicitly providing `-j <N>` to
`ninja` should not cause any problems.

### **Install**

In most cases, installing Kitsune is optional. Unless `KITSUNE_SYSROOT` was
provided, Kitsune can be run from the build directory. It can be more
more convenient to "install" Kitsune to `${CMAKE_INSTALL_PREFIX}`, especially
if `${CMAKE_INSTALL_PREFIX}/bin` is in the `$PATH` environment variable.

```
ninja install
```

On the other hand, if `KITSUNE_SYSROOT` was set, running the Kitsune drivers,
`kitcc`, `kit++` or `kitfc`, from the build directory will require
`--sysroot` to be passed every time. This can be very inconvenient, especially
when compiling non-trivial applications. In this case, installation is generally
required. See [post-install](#post-install) section for more
information.

### **Post-install**

If either `-DKITSUNE_GCC_INSTALL_DIR` or `-DKITSUNE_SYSROOT` were provided
when building Kitsune, it is very likely that the corresponding
`--gcc-install-dir=` and `--sysroot=` options have to be provided when compiling
code with Kitsune. The most convenient method to ensure that these options are
always added is to use [configuration files](ConfigurationFiles.md). These
files must be created and installed manually. Kitsune's build system and tools
do not provide any support for creating, editing or installing these files.
