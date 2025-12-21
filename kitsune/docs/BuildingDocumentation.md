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

to [cmake](https://cmake.org). This will enable generation all documentation
sets. Fine-grained control over which documentation is generated is also
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

If you are only interested in building the API documentation, use the following
configuration option:

```
-DKITSUNE_ENABLE_DOXYGEN=ON
```

The API reference generated in this case will only cover the code in the
`kitsune/` directory. However, most of Kitsune is an extension of LLVM. In
some cases, it may be useful to build LLVM's API reference as well since this
will ensure that documentation for LLVM classes used by Kitsune is reachable
from the Kitsune-specific API reference. This must be enabled explicitly as
follows:

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
been enabled, the `kitsune-docs` will always be available and will only build
those documentation sets that have been enabled. In most cases, therefore, it
is sufficient to run the following command to build the documentation

```
ninja kitsune-docs
```

## Installing

Installing the documentation will be performed alongside a regular install.

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

## Serving

Installing the documentation is recommended, even if you are only interested
in browsing it locally. While it is possible to [serve these from the build
directory](browse-from-build-directory), there are some limitations to the
approach.

The most straightforward way to view the installed user and developer guides
is to open the relevant HTML file in a web browser directly. To open the main
page of the HTML documentation, type the following in a browser's address bar.

```
file:///path/to/kitsune-prefix/share/doc/kitsune/www/index.html
```

Here, `/path/to/kitsune-prefix` is the value `${CMAKE_INSTALL_PREFIX}` was set
to when configuring Kitsune (or the platform-specific
[default value](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)
if an explicit value was not provided).

You could also run a web server locally to serve the pages.
If you do not already have a web server set up, Python provides an easy-to-use,
basic HTTP server. The command below will start a web server that will run
until the process is terminated.

```
python -m http.server -d /path/to/kitsune-prefix/share/doc/kitsune/www
```

In order to view the documentation, type the following in the address bar of
your web browser.

```
localhost:8000/
```

Python's HTTP server listens for requests on port 8000 by default. It can be
[configured](https://docs.python.org/3/library/http.server.html#command-line-interface)
to listen on a different port.

(browse-from-build-directory)=
### Browsing Documentation Without Installing

It is possible to browse the documentation directly from the build directory
with some limitations. The user and developer guides can be found in
`/path/to/kitsune-build/tools/kitsune/docs/html`. Here `/path/to/kitsune-build`
is the absolute path to the directory in which Kitsune is built.

Similar to the approach described above, a web server that serves this
documentation can be started as follows:

```
python -m http.server -d /path/to/kitsune-build/tools/kitsune/docs/html
```

The main page of this documentation can also be reached by typing the following
into the address bar of a web browser.

```
file:///path/to/kitsune-build/tools/kitsune/docs/html/index.html
```

With this approach, the doxygen-generated API reference, even if built, will
not be reachable from the landing page in
`/path/to/kitsune-build/tools/kitsune/docs/html/index.html`. The API reference
will be linked correctly only when the documentation is installed.

To view the main page of the API reference, type the following into your
browser's address bar.

```
file:///path/to/kitsune-build/tools/kitsune/docs/doxygen/html/index.html
```

Or start a web server to serve files directly from the build directory/

```
python -m http.server -d /path/to/kitsune-build/tools/kitsune/docs/doxygen/html
```
