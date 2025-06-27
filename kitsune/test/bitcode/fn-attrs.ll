; Check that the bitcode reader and writer can handle Kitsune-specific function,
; attributes.
;
; RUN: llvm-as %s -o - \
; RUN:     | llvm-dis -o - \
; RUN:     | FileCheck %s
;
; CHECK-DAG: define void @fkernel() #[[KERNEL:[0-9]+]]
; CHECK-DAG: define void @fdevice() #[[DEVICE:[0-9]+]]
;
; CHECK-DAG: #[[KERNEL]] = { kit_kernel }
; CHECK-DAG: #[[DEVICE]] = { kit_device }

define void @fkernel() #3 {
  ret void
}

define void @fdevice() #4 {
  ret void
}

attributes #3 = { kit_kernel }
attributes #4 = { kit_device }
