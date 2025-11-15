The tests here are intended to test most things in Kitsune. Some tests for
Kitsune-specific features may, nevertheless, be in `clang/test` or `llvm/test`,
but most test should be added here.

***General Guidelines***

In general, the use of high-level i.e. source code inputs should be avoided
when not specifically checking the frontend i.e. the command line option
handling, parser, semantic analysis etc. If one wishes to check the effect of
a command line option on the transformation passes, it is better to write
several separate tests such as:

  1. A test that the command line option is handled correctly i.e. that it is
     passed to the driver (`-cc1`, `-fc1`)

  2. A test that checks that the fields of `TTOptions` object are set correctly
     depending on the option (if applicable)

  3. A test of the behavior of one or more of the transformation passes when
     the option is set. Because of the way Kitsune is designed, the frontend
     options will usually have a corresponding option that can be used directly
     with `opt`.

Doing so ensures that every part of the compiler that has been modified when
adding a command line option is tested.

It is also often easier to craft some LLVM-IR by hand to check a specific code
path in an analysis or optimization pass than it is to build a high-level
test case that will be lowered to comparable IR. In such cases, the high-level
source code may necessarily have to be fairly complex in order for it to get
through the various standard passes (mem2reg, instcombine, adce etc.) in the
desired form.

***Organization***

The tests are organized into directories and subdirectories. The top-level
directories include:

  - `analysis`: Tests for Kitsune-specific analysis passes. These may include
    analysis passes that operate on embedded modules. If the analysis is very
    specific to a tapir target, it may be better to add the test to
    `tapir/<tapir-target>` instead.

  - `assembler/`: Checks that Kitsune-specific attributes and other constructs
    are parsed correctly from human-readable LLVM assembly.

  - `bitcode/`: Checks that Kitsune-specific attributes and other constructs
    are parsed from LLVM bitcode correctly.

  - `driver/`: Check basic handling of Kitsune-specific command line options,
    especially those that apply to all tapir targets. This should only check
    that the behavior in the frontend driver is as expected. Tests that check
    the effect of a command line option on analyses, optimization and code
    generation should be added to the directories for specific tapir targets or
    frontends.

  - `frontend/`: This contains tests for the supported language-specific
    frontends, `kitcc`, `kit++`, `kitfc` etc. Some of the features that these
    are intended to check include:

      - Language-specific builtins

      - Optimization defaults, in particular where they differ from the
        defaults in `clang` and `flang`. For instance, the default value of
        the `fp-contract` command line option in `kitcc` and `kit++` differs
        from that in `clang`.

  - `lang/`: This contains tests that test the top-level language constructs
    added by Kitsune. These primarily are intended to check that the parsing,
    semantic analysis and LLVM-IR generation are as expected. The tests in this
    directory may be subdivided into tests for specific constructs. For
    instance, `lang/` currently contains `forall/` to check handling of the
    `forall` construct, `attr/` to check handling of Kitsune-specific
    attributes etc.

  - `plugins/`: Tests for the plugins that can be used with Kitsune. This
     includes both tapir target plugins and pass plugins.

  - `tapir/`: This contains tests for the behavior of specific tapir targets.
    This is subdivided into directories for each tapir target. The tests within
    the tapir-target-specific subdirectories check everything from command
    line option handling to device-specific code generation (in the case of
    GPU-centric tapir targets for instance which have to generate code for
    both host and device).

  - `tools/`: Tests for Kitsune-specific tools such as `kit-mbc`, `kit-config`
    etc. Each tool will have its own subdirectory. The LLVM subdirectory
    contains tests of LLVM tools that have been modified with Kitsune-specific
    features. For example, `opt` and `llc` both have been customized with
    special handling of Kitsune's passes.

  - `transforms/`: Tests for passes whose behavior is not very closely tied to
    a given tapir target. These are typically passes that operate on the host
    module, but they need not be. In some cases, the tests may have different
    checks depending on the tapir target that may have been specified. There
    aren't any strict guidelines on whether such tests should be added to the
    tests for a specific tapir target in `tapir/` or added here. Exercise your
    judgment.

  - `verifier/`: This intended to check that the verifier works correctly with
    modules generated by Kitsune. The verification checks are integrated into
    core LLVM (in `llvm/lib/IR/Verifier.cpp`), but are nevertheless tested here
    since the additions are Kitsune-specific.
