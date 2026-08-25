// REQUIRES: kitsune-examples
//
// Check that the options specific to the pthreads tapir target make it to the
// tapir target options. We require examples to be built because this test
// requires a valid tapir target plugin to work correctly
//
// RUN: %kitxx --tapir=custom --tapir-plugin=%kit-tt-plugin-demo %s \
// RUN:     -S -emit-llvm -o /dev/null -O2 \
// RUN:     -mllvm -print-tt-options 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ALL
//
// ALL: Tapir target options
// ALL: Primary: custom
// ALL: Custom plugin: TTPluginDemo 1.0
// ALL: Custom plugin file: {{.+}}/kit-tt-plugin-demo.{{.+}}
