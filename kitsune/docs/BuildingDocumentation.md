# Building Kitsune's Documentation

Kitsune documentation sets include usage manuals, developer guides,
Doxygen-based documentation of the internal data structures and APIs, and man
pages. The usage manuals and developer guides consist of markdown files that are
translated to HTML suitable for a browser, the API reference is generated from
the source code as a collection of HTML files, while the man pages are intended
for use in a terminal. Here, we describe how to build each of these
documentation sets.


## Prerequisites

A list of the tools required to build all of Kitsune's documentation sets
can be found
{{'[here](https://{}/{}/kitsune/tree/{}/kitsune/docs/requirements.txt)'.format(kitsune_repo_host, kitsune_repo_owner, kitsune_repo_branch)}}.

```{note}
The versions of the tools in the requirements file are the oldest versions with
which the documentation has been built. In some cases, older versions may also
work, but that is not guaranteed.
```

The table below lists the prerequisites required to generate each documentation
set. Only the prerequisites for the specific documentation sets that are
enabled must be present at configure-time.

```{table}
| Documentation | Requirements |
| :-----------: | :----------- |
| API Reference | [doxygen](https://www.doxygen.nl), [graphviz](https://graphviz.org) |
| Usage and Design | [sphinx](https://www.sphinx-doc.org), [myst-parser](https://myst-parser.readthedocs.io) |
| Man pages | sphinx, myst-parser |
```

On Linux, these should be readily available in the package managers of most
major distributions. Alternatively, Python's pip tool can be used to obtain
sphinx and myst-parser.


## Configuration

Kitsune's documentation is not built by default and must be enabled explicitly
when [configuring](GettingStarted.md#configure) Kitsune. This requires passing
the configuration option:

```
-DKITSUNE_BUILD_DOCS=ON
```

to [CMake](https://cmake.org). This will enable generation of all documentation
sets. Fine-grained control over which documentation set is generated is also
possible.

If you are _not_ interested in the API documentation, set
`KITSUNE_ENABLE_DOXYGEN` to `OFF`, and use the following configuration option
instead.

```
-DKITSUNE_ENABLE_SPHINX=ON
```

```{note}
It is not currently possible to build the man pages without building the HTML
documentation.
```

If you are _only_ interested in building the API documentation, use the
following configuration option:

```
-DKITSUNE_ENABLE_DOXYGEN=ON
```

The API reference generated in this case will only cover the code in the
`kitsune/` directory. However, most of Kitsune is an extension of LLVM. It may,
therefore, be useful to build LLVM's API reference as well. This would ensure
that documentation for LLVM classes used by Kitsune is reachable from the
Kitsune-specific API reference. This must be enabled explicitly as follows:

```
-DKITSUNE_ENABLE_LLVM_DOXYGEN=ON
```

The API reference documentation requires more than 2GB of disk space in the
build directory. Installing it will require another 2GB. Generating the
documentation could take several minutes depending on the number of CPU cores
available on the system.

```{warning}
When explicitly setting `KITSUNE_ENABLE_DOXYGEN` or `KITSUNE_ENABLE_SPHINX`,
**_do not_** set `-DKITSUNE_BUILD_DOCS=ON`. Doing so will result in both
`KITSUNE_ENABLE_DOXYGEN=ON` and `KITSUNE_ENABLE_SPHINX=ON`.
```

The listing below is the minimal number of configuration options needed to
build Kitsune's documentation. Clearly, it will enable both the Doxygen-based
API reference and the HTML documentation.

```shell
cmake -G Ninja \
      -DKITSUNE_BUILD_DOCS=ON \
      /path/to/kitsune/llvm
```


## Building

Even if one or more of the documentation sets are enabled at configure-time,
they will _not_ be built automatically. Instead, `ninja` (or `make` if the
`Unix Makefiles` generator was specified) must be run with one or more of the
documentation-specific targets summarized in the table below.

```{table}
| Target | Documentation Set |
| :----: | :---------------- |
| `kitsune-docs` | Build all enabled documentation sets |
| `kitsune-docs-doxygen` | API reference (including LLVM's API reference if enabled) |
| `kitsune-docs-html` | User and developer guides only |
| `kitsune-docs-man` | Manual pages only |
```

Depending on the configuration options that were set, not all targets in the
table above will be available. Note that as long as building documentation has
been enabled, the `kitsune-docs` target will always be available and will build
all documentation sets that have been enabled. In most cases, therefore, it
is sufficient to run the following command to build the documentation.

```
ninja kitsune-docs
```

When building the HTML documentation with `kitsune-docs-html`, the Doxygen
documentation will also be built (as long as it has been enabled). To browse the
HTML documentation, therefore, it is sufficient to use

```
ninja kitsune-docs-html
```

The reverse does not hold. That is, if the following command is used

```
ninja kitsune-doxygen
```

the Doxygen-generated API reference will be built, but the HTML documentation
will not be built.


## Installing

Installation of the documentation, on the other hand, will be performed
automatically with the regular install target.

```
ninja install
```

Currently, there is no way to install individual documentation sets.

```{important}
The documentation must have been [built](#building) before installing. Failure
to do so will result in an error.
```

The HTML documentation, that is, the user and developer guides and the
doxygen-generated API documentation will be installed to
`${CMAKE_INSTALL_PREFIX}/share/doc/kitsune/www`.

```{note}
Kitsune's documentation is not installed to the same directory as the
documentation of other LLVM projects. This is intentional.
```

The man pages, on the other hand, are installed to the same location as the
man pages for LLVM's frontends, `clang` and `flang`.


## Browsing

The generated documentation can be browsed from the build directory as well as
the install directory. The root directory containing the HTML documentation in
the build directory is:

```
/path/to/kitsune-build/tools/kitsune/docs/html
```

Here, `/path/to/kitsune-build` is the top-level directory in which Kitsune is
configured and built. In the commands below, this is referred to as
`${DOCROOT}`.
The most straightforward way to view the installed user and developer guides
is to open the relevant HTML file in a web browser directly. To open the main
page of the HTML documentation, type the following in a browser's address bar.

```
file://${DOCROOT}/index.html
```

A web server could also be run locally to serve the pages. If one is not already
set up, Python provides an easy-to-use, basic HTTP server. The command below
will start the web server.

```
python -m http.server -d ${DOCROOT}
```

One it has started, type the following into the address bar of a web browser.

```
http://localhost:8000/
```

This should serve up the main page of Kitsune's documentation.
Python's HTTP server listens for requests on port 8000 by default. It can be
[configured](https://docs.python.org/3/library/http.server.html#command-line-interface)
to listen on a different port instead.

These methods also apply to the installed documentation. In this case, replace
`${DOCROOT}` in the commands above with
`${CMAKE_INSTALL_PREFIX}/share/doc/kitsune/www`.

```{important}
Before browsing the HTML documentation, ensure that at least `kitsune-docs-html`
has been run.
```
