# Kitsune's Documentation

TODO: Write a reasonable description

```{warning}
Kitsune is under active development. While we make every effort to
keep this documentation up-to-date, parts of it may not reflect the current
state of Kitsune. In such cases, the inline documentation in Kitsune's source
code is more likely to be accurate.
```

The user guides are intended for users of Kitsune, while the developer guides
are intended for those interested in contributing to Kitsune's development. The
design documents describe some aspects of Kitsune's design though they are not,
and may never be, comprehensive.

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
    FAQ
```

# References

```{eval-rst}
.. toctree::
    :titlesonly:

    KitClangOptionsDoc
    KitFlangOptionsDoc
    API Reference<doxygen/index.md>
    KitAttrsDoc
    KitLLVMAttrsDoc
    KitInstructionsDoc
    KitIntrinsicsDoc
    KitPassesDoc
    CommandGuide/index.md
    Glossary
```

# Design Documents

```{eval-rst}
.. toctree::
    :titlesonly:

    BuildSystem
    CodeOrganization
    DriverDesign
    EmbeddedBitcode
    PassPipeline
```

# Developer Guides

```{eval-rst}
.. toctree::
    :titlesonly:

    AddingCommandLineOption
    AddingTapirTarget
    AddingKitsuneIntrinsic
    BuildingDocumentation
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
