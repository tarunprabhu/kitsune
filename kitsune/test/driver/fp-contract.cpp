// -----------------------------------------------------------------------------
// The default FP contract value is different from clang's defaults. The new
// default is only used if a kitsune frontend is used.
//
// -----------------------------------------------------------------------------
// Check that the defaults have not changed when running without a Kitsune
// frontend.
//
// RUN: %clangxx -### %s 2>&1 | FileCheck %s -check-prefix CONTRACT-ON
//
// -----------------------------------------------------------------------------
// When running with a Kitsune frontend, this value should be ON. This is
// mainly relevant when compiling cuda/hip code where clang's defaults are not
// on. But that is tested elsewhere.
//
// RUN: %kitxx -### %s 2>&1 | FileCheck %s -check-prefix CONTRACT-ON
//
// CONTRACT-ON: "-ffp-contract=on"
