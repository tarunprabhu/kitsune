# Configuration Files

Like clang and flang, Kitsune supports configuration files that can be used to
automatically add command-line options when invoking the compiler. This is
particularly useful when running Kitsune on systems with a non-standard sysroot,
or where the headers and libraries provided by the system are old and a more
modern toolchain has to be installed elsewhere. Note that configuration files
are always optional - their absence will never result in any errors or warnings.

Clang's
[documentation](https://clang.llvm.org/docs/UsersManual.html#configuration-files)
describes how to use configuration files. Kitsune follows the rules described
there exactly. This document summarizes some of the information from Clang's
documentation and only describes basic usage - which should be sufficient for
most users that need to use configuration files.

In addition, Kitsune supports some additional functionality that will also be
[described](#configuration-files-for-tapir-targets).

## Basic Usage

As described in Clang's
[documentation](https://clang.llvm.org/docs/UsersManual.html#configuration-files),
the name of the configuration file must follow a specific format. For Kitsune's
drivers, these are listed in the table below. The file names for each driver
are listed from top to bottom in the order in which they will be searched.
For instance, for `kitcc`, Kitsune will first search for a file named in
`<triple>-kitcc.cfg` in the standard search directories as described in the
documentation linked above. If this is not found, a file named `kitcc.cfg` will
be searched for followed by `<triple>.cfg`. Here, `<triple>` is the target
triple option that is specified when compiling with Kitsune. If an explicit
target triple is not specified, the default target triple set when building
Kitsune will be used.
Unless Kitsune has been explicitly built as a cross-compiler [^1], this will
be the target triple of the host system.

[^1]: While using Kitsune as a cross-compiler should work, this is not - and is unlikely to ever be - a supported use case

```{table}
| Driver | Use | Config File Names |
| :----: | :-- | :---------------: |
| `kitcc` | For C programs | `<triple>-kitcc.cfg`<br>`kitcc.cfg`<br>`<triple>.cfg` |
| `kit++` | For C++ programs | `<triple>-kit++.cfg`<br>`kit++.cfg`<br>`<triple>.cfg` |
| `kitfc` | For Fortran programs | `<triple>-kitfc.cfg`<br>`kitfc.cfg`<br>`<triple>.cfg` |
```


In order to install a configuration file for `kit++`, one could create a file
named `kit++.cfg` in the same directory as the `kit++` executable. If the
contents of `kit++.cfg` are as follows

(configuration-files-example)=
```
--sysroot=/path/to/non/standard/sysroot
-Wl,--dynamic-linker=/path/to/some/other/ld.so
```

the simple invocation of `kit++` below

```
kit++ -O1 --tapir=cuda in.c
```

would be as if one had explicitly invoked the following

```
kit++ --sysroot=/path/to/non/standard/sysroot -Wl,--dynamic-linker=/path/to/some/other/ld.so -O1 --tapir=cuda in.c
```

## Configuration files for tapir targets

Kitsune supports configuration files specific to each tapir target [^2]. These
files are only used when the corresponding tapir target is requested with the
`--tapir=` option when invoking Kitsune. The
tapir targets that support configuration files and the required names of these
files are listed in the table below:

```{table}
| Tapir Target | Config File Name |
| :----------: | :--------------: |
| [cuda](tapir-targets-cuda) | cuda.cfg |
| [hip](tapir-targets-hip) | hip.cfg |
| [nolo](tapir-targets-nolo) | nolo.cfg |
| [opencilk](tapir-targets-opencilk) | opencilk.cfg |
| [pthreads](tapir-targets-pthreads) | pthreads.cfg |
| [serial](tapir-targets-serial) | serial.cfg |
```

Like the driver-specific configuration files, these must be present in either
the user configuration directory, the system configuration directory, or the
directory in which the Kitsune driver executables reside. See Clang's
[documentation](https://clang.llvm.org/docs/UsersManual.html#configuration-files)
for information on the default locations of these directories and how to set
them when configuring Kitsune. As with the driver-specific configuration files,
the tapir-target-specific configuration files are optional.

Note that, unlike the driver-specific configuration files, a variant with a
`<triple>` is not supported.

For example, in order to specify additional options that should always be used
with the [hip](tapir-targets-hip) tapir target, create a file named `hip.cfg` in
the directory containing the Kitsune driver executables. If the contents of
`hip.cfg` is as shown below,

```
--tapir-hip-sramecc=any
--tapir-hip-xnack=any
```

the simple invocation of `kitcc` below,

```
kitcc --tapir=hip -O2 in.c
```

will be equivalent to

```
kitcc --tapir-hip-sramecc=any --tapir-hip-xnack=any --tapir=hip -O2 in.c
```

If the value of the `--tapir=` is not `hip`, `hip.cfg` will _**not**_ be used.

If a configuration file for `kitcc.cfg` is also present, it will be used in
addition to any tapir-target-specific configuration files. For instance, if the
contents of `kitcc.cfg` are the same as the example
[shown earlier](configuration-files-example), the simple invocation of `kitcc`
with `--tapir=hip` above will be equivalent to

```
kitcc --tapir-hip-sramecc=any --tapir-hip-xnack=any \
      --sysroot=/path/to/non/standard/sysroot -Wl,--dynamic-linker=/path/to/some/other/ld.so \
      --tapir=hip -O2 in.c
```

The example above has been split across multiple lines for clarity.

## Example of configuration files in a custom directory

```{attention}
This has not been written because there is currently a Kitsune-specific
`--config-kitsune-dir` option. This might be removed in favor of just using
`--config-user-dir` that is already present in clang. This is being discussed
by the Kitsune development team.
```

TODO: Show an example of installing configuration files in a custom directory
i.e. one that is not provided when building Kitsune. This can be useful when
Kitsune is installed in a location that is not writable

[^2]: Configuration files are **_not_** supported by the [custom](tapir-targets-custom) tapir target.
