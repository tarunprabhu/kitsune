# Building Kitsune's Documentation

Kitsune contains HTML documentation targeted at both users and Kitsune's
developers as well as Doxygen-based API documentation of the internal data
structures and API's. This document provides more information on how to build
and (locally) serve this documentation.

## Prerequisites

A complete list of the tools required to build Kitsune's documentation can be
found
{{'[here](https://{}/{}/kitsune/tree/{}/llvm/docs/requirements.txt)'.format(kitsune_repo_host, kitsune_repo_owner, kitsune_repo_branch)}}.
This list includes prerequisites for building both the API and HTML
documentation. The table below lists the requires for each form of documentation
separately

```{table}
| API | HTML |
| :-: | :--: |
| doxygen | sphinx |
| docutils | myst-parser |
```
On Linux, these should be readily available in the package managers of most
major distributions.

## Configuration

Kitsune's documentation is not built by default and must be enabled explicitly
when [configuring](GettingStarted.md#configure) Kitsune. This requires passing
the configuration option

```
-DKITSUNE_BUILD_DOCS=ON
```

to [cmake](https://cmake.org). In this case, both the API and HTML documentation
will be enabled. This means that the prerequisites for both the API and
HTML documentation must be satisfied at configure-time. If any are not,
configuration will fail. Fine-grained control over exactly which documentation
is enabled is provided.

If you are only interested in building the API documentation, and not the HTML
documentation, use the following configuration option:

```
-DKITSUNE_ENABLE_DOXYGEN=ON
```

If you are only interested in building HTML documentation, and not the API
documentation, use the following configuration option:

```
-DKITSUNE_ENABLE_SPHINX=ON
```

```{warning}
When explicitly setting `KITSUNE_ENABLE_DOXYGEN` or `KITSUNE_ENABLE_SPHINX`,
**_do not_** set `-DKITSUNE_BUILD_DOCS=ON`. Doing so will result in both
`KITSUNE_ENABLE_DOXYGEN=ON` and `KITSUNE_ENABLE_SPHINX=ON`.
```

## Building

The targets to build the documentation are not added to the
default build target. Therefore, simply running `ninja` (or `make` if you use
the `Unix Makefiles` generator) is not sufficient. To build all of Kitsune's
documentation, run the following in the build directory

```
ninja kitsune-docs
```

This will build all documentation that has been enabled, and will omit those
that have been disabled. For example, if `-DKITSUNE_BUILD_DOCS=ON` was
specified at configure-time, both API and HTML documentation will be built.
On the other hand, if only `-DKITSUNE_BUILD_DOXYGEN=ON` was specified, only the
API documentation will be built.

## Installing

It is not strictly necessary to install Kitsune in order to access the
documentation. However, it may sometimes be convenient to do so, especially if
you are interested in accessing Kitsune's man pages and have configured
`man` to search specific directories. Installing the documentation will be
performed alongside a regular install

```
ninja install
```

## Serving

The simplest way to check the HTML documentation is to simple open the relevant
HTML file in a web browser. For example, the following command will open the
landing page of Kitsune's HTML documentation in
[Firefox](https://www.firefox.com).

```
firefox /path/to/kitsune-build/tools/kitsune/docs/html/index.html
```

Here, `/path/to/kitsune-build` is the absolute path to the top-level directory
in which Kitsune was configured and built. Alternatively, you can type the
following in the address bar of your web browser.

```
file:///path/to/kitsune-build/tools/kitsune/docs/html/index.html
```

You may wish to run a web server locally to check that the pages
are served as expected. If you do not already have a web server setup,
a fairly straightforward approach would be to use
Python's HTTP server. This should already be available on most systems. The
command below will serve files from the build directory directly. This is
convenient since you do not need to install the documentation in order to view
it.

```
python -m http.server -d /path/to/kitsune-build/tools/kitsune/docs/html 8080
```

This will start a web server that will stay active until the process is
terminated. In order to view the documentation, type the following in the
address bar of your web browser.

```
localhost:8080/
```

You should now see the landing page of the HTML documentation.

The port on which web server listens for requests can be changed. In the command
that starts the webserver, simply replace 8080 with the desired port number.
For example, if you specify 12345 as the port when starting the web server,
navigate to `localhost:12345/` in your web browser to view the built
documentation.
