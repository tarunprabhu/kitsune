; REQUIRES: kitsune-examples
;
; RUN: opt --tapir=custom --tapir-plugin=%kit-ttplugin-demo %s \
; RUN:     -o /dev/null -O2 \
; RUN:     -dump-tapir-target-options 2>&1 \
; RUN:     | FileCheck %s --check-prefix=TTO
;
; TTO: Custom plugin: TTPluginDemo 1.0
; TTO: Custom plugin file: {{.+}}/kit-ttplugin-demo.{{.+}}
