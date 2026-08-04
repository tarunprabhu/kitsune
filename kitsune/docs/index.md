# The Kitsune Compiler Documentation

```{warning}
Kitsune is under active development. While we make every effort to
keep this documentation up-to-date, parts of it may not reflect the current
state of Kitsune. In such cases, the inline documentation in Kitsune's source
code is more likely to be accurate.
```

This is a loosely connected collection of pages that describe the Kitsune
compiler, it's design and usage. The pages are grouped into collections that are
likely to be useful for different people. The user guides are intended for users
of Kitsune, this is those who are interested in using Kitsune to compile their
code, but are not necessarily interested in how it works, or in contributing to
its development. The developer guides
are intended for those interested in contributing to Kitsune's development. The
design documents describe some aspects of Kitsune's design though they are not,
and may never be, comprehensive.

The [overview](Overview.md) page provides a (very) high-level description of
Kitsune and it's use. This is where most readers, especially those unfamiliar
with Kitsune should probably start. Everything else can be read in any order
depending on what is of interest to you.


# User Guides

```{eval-rst}
.. toctree::
    :titlesonly:

    Overview
    TapirTargets
    GettingStarted
    BasicUsage
    LanguageExtensions
    MemoryManagement
    FortranSupport
    KokkosSupport
    ConfigurationFiles
    StaticLinking
    Instrumentation
    RuntimeEnvVar
    FAQ
```

# References

<!--
    Using inline HTML here because the target of the link changes depending on
    whether or not Doxygen documentation has been enabled. The Doxygen
    documentation is built directly in the output directory, but nothing there
    can be referenced from here.
-->
{{'<ul><li><a href="{}">API Reference</a></li></ul>'.format(kitsune_api_reference_link)}}

```{eval-rst}
.. toctree::
    :titlesonly:

    KitClangOptionsDoc
    KitCC1OptionsDoc
    KitFlangOptionsDoc
    KitFC1OptionsDoc
    KitAttrsDoc
    KitCBuiltinsDoc
    KitArgAttrsDoc
    KitFuncAttrsDoc
    KitGVAttrsDoc
    KitInstAttrsDoc
    KitInstructionsDoc
    KitIntrinsicsDoc
    KitLoopAttrsDoc
    KitModuleAttrsDoc
    KitPassesDoc
    TapirPassesDoc
    CommandGuide/index.md
    Glossary
```

# Design Documents

```{eval-rst}
.. toctree::
    :titlesonly:

    BuildSystem
    CodeOrganization
    CommandLineOptions
    DriverDesign
    EmbeddedBitcode
    InstrumentationDesign
    PassPipeline
    LTO
```

# Developer Guides

```{eval-rst}
.. toctree::
    :titlesonly:

    AddingCommandLineOption
    AddingTapirTarget
    AddingKitsuneIntrinsic
    AddingSharedType
    BuildingDocumentation
    LLVMIRAttributes
    KitsuneTestSuite
    Testing
    LLVMTools
    RegisteringLibraryFunction
    WritingEmbeddedBitcodePass
    WritingPassPlugin
    WritingTapirTargetPlugin
```

# Release Notes

```{eval-rst}
.. toctree::
    :titlesonly:

    ReleaseNotes
```

# Indices

```{eval-rst}
* :ref:`genindex`
* :ref:`search`
```
