; Check that the bitcode reader and writer can handle Kitsune-specific global
; variable attributes.
;
; The command below does the following:
;
;   - Generate a file containing global variables for the embedded bitcode and
;     fat binary. The embedded bitcode is generated from this file
;   - Rename these globals to ensure that the reader does not expect specific
;     names for these globals
;   - Add a global variable that contains properties for the kernel
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | sed -E $'/^[@][.]kit[.]emb[.]fb/a\\\n \
; RUN:         @kp = constant {i64, i64} zeroinitializer #2' \
; RUN:     | sed -E $'/^attributes #1/a\\\n \
; RUN:         attributes #2 = { kit_tt(4) "kit_kernel_props"="kit_kernel" }' \
; RUN:     | sed 's/[.]kit[.]emb[.]//g' \
; RUN:     | llvm-as -o - \
; RUN:     | llvm-dis -o - \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @bc = {{.+}} #[[BC:[0-9]+]]
; CHECK-DAG: @fb = {{.+}} #[[FB:[0-9]+]]
; CHECK-DAG: @kp = {{.+}} #[[PROPS:[0-9]+]]
;
; CHECK-DAG: #[[BC]] = { kit_bc kit_tt(4) }
; CHECK-DAG: #[[FB]] = { kit_fb kit_tt(4) }
; CHECK-DAG: #[[PROPS]] = { kit_tt(4) "kit_kernel_props"="kit_kernel" }

define void @kit_kernel() {
  ret void
}
