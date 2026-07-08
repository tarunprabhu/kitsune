# Unit tests for Kitsune's runtime

This `common/` subdirectory contains unit tests for the parts of the runtime
that are shared between the tapir-target-specific runtimes. Unit tests for
the tapir-target-specific runtimes should be added to subdirectories with the
same name and casing as the corresponding tapir targets, `cuda`, `openmp`,
`serial`, etc. The tests for the tapir-target-specific runtimes should be
built only if the corresponding tapir target has been enabled.
