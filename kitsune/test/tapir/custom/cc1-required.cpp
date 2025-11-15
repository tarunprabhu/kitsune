// Check that an error is emitted if any of the required options are not
// provided
//
// RUN: not %kitxx -cc1 --tapir=custom %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=MISSING_PLUGIN
//
// MISSING_PLUGIN: missing required option '--tapir-plugin='
