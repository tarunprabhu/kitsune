; The tapir target id in the module metadata of embedded modules must be valid
;
; RUN: %kit-enc --tapir=hip --skip-metadata %s \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: embedded module requires valid tapir target in device module metadata

!kitsune.device.module.flags = !{!0, !1}

!0 = !{i32 -200000000}
!1 = !{!"some-silly-name"}
