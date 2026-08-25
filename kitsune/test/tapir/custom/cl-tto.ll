; REQUIRES: kitsune-examples
;
; RUN: opt --tapir=custom --tapir-plugin=%kit-tt-plugin-demo %s \
; RUN:     -passes=kit-print-tt-options -disable-output \
; RUN:     | FileCheck %s
;
; CHECK: Tapir target options:
; CHECK: Primary: custom
; CHECK: Custom plugin: TTPluginDemo 1.0
; CHECK: Custom plugin file: {{.+}}/kit-tt-plugin-demo.{{.+}}
