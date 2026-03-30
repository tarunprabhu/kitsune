; The tapir target id in the module metadata of embedded modules must be valid
;
; RUN: %kit-enc --tapir=hip --skip-metadata %s \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: attribute 'kit.module.device.module.flags': invalid value at index '0'
; CHECK-SAME: Tapir target does not generate embedded bitcode
; CHECK-NEXT: embedded module: broken module found

!kit.module = !{!0}

!0 = distinct !{!0, !1}
!1 = !{!"kit.module.device.module.flags", i32 1, !"gallumbits"}
