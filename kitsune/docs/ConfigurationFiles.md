# Configuration Files

Like [clang](https://clang.llvm.org) and [flang](https://flang.llvm.org),
Kitsune supports configuration files that can be used to
automatically add command-line options when invoking the compiler. This is
particularly useful when running Kitsune on systems with a non-standard sysroot,
or where the headers and libraries provided by the system are old and a more
modern toolchain has to be installed elsewhere.

Kitsune supports two categories of configuration files:

- [**Driver-specific configuration files**](configuration-files-for-drivers):
  These are used by Kitsune's drivers - `kitcc`, `kit++`, and `kitfc`.

- [**Tapir-target-specific configuration files**](configuration-files-for-tapir-targets):
  These are used by specific tapir targets. These are only read when the
  corresponding `<tapir-target>` is provided to the `--tapir=` command-line
  option.

(configuration-files-for-drivers)=
## Configuration Files for Drivers

Configuration files for Kitsune's drivers may be specified explicitly on the
command-line, or loaded from default locations. If an appropriate file is
present in the default location, and one has been specified explicitly, the
default configuration file is loaded first.

An example of a configuration file is shown below.

(configuration-files-example)=
```
--sysroot=/path/to/non/standard/sysroot
-Wl,--dynamic-linker=/path/to/ld.so
```

Here, `--sysroot=` is a compiler option and will be passed to the frontend
compiler. There also a linker option that will be passed to the linker.

If the configuration file provided above is loaded, the invocation below

```
kit++ ...
```

will be equivalent to the following

```
kit++ --sysroot=/path/to/non/standard/sysroot -Wl,--dynamic-linker=/path/to/ld.so \
      ...
```

There are several ways that a configuration file can be loaded. Each of these
is described in the sections below.

### Explicit Configuration Files

A configuration file can be specified explicitly with the `--config`
command-line option. If the option is specified more than once, all specified
files are loaded in the order in which they appear on the command-line.

```
kit++ --config=/path/to/config.cfg ...
```

If the value of the `--config` option contains a directory separator, it is
considered a file path, and options are read from that file. Otherwise the
value is treated as a file name. A file with this name is searched for
sequentially in the following "config directories":

- User directory
- System directory
- Directory containing the Kitsune driver executable

The user and system directories for the configuration files can be specified
at runtime using the `--config-user-dir` and `--config-system-dir` command-line
options respectively.

Alternatively, these can be set at configure-time using, respectively, the
`KITSUNE_CONFIG_FILE_USER_DIR` and `KITSUNE_CONFIG_FILE_SYSTEM_DIR`
configure-time options. Specifying config directories on the command-line will
override the corresponding directories set at configure-time.

```{warning}
If a config file name is provided in the `--config` option, it is an error if a
file with that name is not found in the default - or explicitly specified -
directories.
```

### Default Configuration Files

The default configuration files will be searched for sequentially in the
following directories:

- User directory
- System directory
- Directory containing the Kitsune driver executable

Note that these are the same directories as described in the
[previous section](#explicit-configuration-files). The table below lists the
name of the configuration file that Kitsune will attempt to load depending on
the driver being used.

```{table}
| Driver | Config File Name |
| :----: | :--------------: |
| `kitcc` | `kitcc.cfg` |
| `kit++` | `kit++.cfg` |
| `kitfc` | `kitfc.cfg` |
```

It is **_not_** an error if a configuration file with the given name is not
found in any of the default directories. This is also the case if an explicit
directory is  specified using either the `--config-user-dir` or
`--config-system-dir` command-line options.

```{tip}
Loading default configuration files can be disabled entirely using the
`--no-default-config` command-line flag.
```

We have only described basic use of default configuration files. This should
be sufficient for most users of Kitsune who need configuration files. For more
advanced options, consult
[clang's documentation](https://clang.llvm.org/docs/UsersManual.html#configuration-files).
Some of the advanced uses involve per-target-triple configuration files. These
are mainly useful when cross-compiling. While it should work as expected,
cross-compiling with Kitsune is not supported.

(configuration-files-for-tapir-targets)=
## Configuration Files for Tapir Targets

Kitsune supports configuration files specific to each tapir target. These
files are only used when the corresponding tapir target is requested with the
`--tapir=` option when invoking Kitsune. The tapir targets that support
configuration files and the required names of these files are listed in the
table below.

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

For example, in order to specify additional options that should always be used
with the [hip](tapir-targets-hip) tapir target, create a file named `hip.cfg` in
one of the directories described below. If the contents of `hip.cfg` is as shown
below,

```
--tapir-hip-sramecc=any
--tapir-hip-xnack=any
```

the following simple invocation of `kitcc`

```
kitcc --tapir=hip ...
```

will be equivalent to

```
kitcc --tapir-hip-sramecc=any --tapir-hip-xnack=any \
      --tapir=hip ...
```

If the value of the `--tapir=` is not `hip`, `hip.cfg` will _**not**_ be used.

```{important}
A tapir-target-specific configuration file cannot be specified for the
[custom](tapir-targets-custom) tapir target. Driver-specific configuration files
will continue to be used when `--tapir=custom` is specified.
```

These files will be searched for in the same directories, and in the same order,
as the driver-specific configuration files described in the
[previous section](#default-configuration-files):

- User directory
- System directory
- Directory containing the Kitsune driver executable

As with those files, the tapir-target-specific configuration files are optional.
Their absence will _not_ raise an error.

```{note}
Unlike the driver-specific configuration files, there is no way to explicitly
provide a path to a specific tapir-target-specific configuration file.
```

These configuration files are always used in addition to any driver-specific
configuration files. For instance, if an appropriate configuration file for
the `kitcc` driver was found, or if one was explicitly provided using the
`--config` command-line option, and its contents are the same as the example
[described earlier](configuration-files-example), then the invocation of
`kitcc` below

```
kitcc --tapir=hip ...
```

will be equivalent to

```
kitcc --tapir-hip-sramecc=any --tapir-hip-xnack=any \
      --sysroot=/path/to/non/standard/sysroot -Wl,--dynamic-linker=/path/to/ld.so \
      --tapir=hip ...
```
