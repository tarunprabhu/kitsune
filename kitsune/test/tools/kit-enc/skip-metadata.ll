; If the --skip-metadata option is passed, the standard device module metadata
; must not be added to the embedded bitcode.
;
; ------------------------------------------------------------------------------
; If the input module does not contain any metadata, nothing will be added.
;
; RUN: %kit-enc --tapir=cuda --skip-metadata %S/input/empty.ll \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s --check-prefix MISSING
;
; MISSING-NOT: kit.device.module.flags
;
; ------------------------------------------------------------------------------
; If the input module already contains metadata, ensure that it is not modified
; in any way, even if it is invalid. In the test below, the metadata name is
; invalid (since an empty string is not allowed) and the tapir target in the
; metadata does not match the tapir target that is specified on the command
; line.
;
; RUN: %kit-enc --tapir=hip --skip-metadata %s \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s --check-prefix RETAIN
;
; RETAIN: !kit.device.module.flags = !{!0, !1}
; RETAIN: !0 = !{i32 2}
; RETAIN: !1 = !{!"gonville-caius"}

!kit.device.module.flags = !{!0, !1}

!0 = !{i32 2}
!1 = !{!"gonville-caius"}
