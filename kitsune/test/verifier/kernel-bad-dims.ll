; The value of the kernel attribute must be in the range [1,3].
;
; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: attribute 'kit.func.kernel': value '0' not in range [1,3]
; CHECK-NEXT: from function 'f0'
; CHECK-NEXT: attribute 'kit.func.kernel': value '4' not in range [1,3]
; CHECK-NEXT: from function 'f4'

define void @f0() !kit.func !0 {
  ret void
}

define void @f1() !kit.func !2 {
  ret void
}

define void @f2() !kit.func !4 {
  ret void
}

define void @f3() !kit.func !6 {
  ret void
}

define void @f4() !kit.func !8 {
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"kit.func.kernel", i32 0}
!2 = distinct !{!2, !3}
!3 = !{!"kit.func.kernel", i32 1}
!4 = distinct !{!4, !5}
!5 = !{!"kit.func.kernel", i32 2}
!6 = distinct !{!6, !7}
!7 = !{!"kit.func.kernel", i32 3}
!8 = distinct !{!8, !9}
!9 = !{!"kit.func.kernel", i32 4}
