; If the tapir target is 'nolo', Kitsune's mobile intrinsics should not be
; lowered. All attributes, calling conventions, and metadata on the call
; instructions should be preserved.
;
; RUN: opt --tapir=nolo -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes CHECK %s

; CHECK-LABEL: @allocate
; CHECK-SAME: i64 %[[N:[^)]+]]
;
; CHECK-NEXT: call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 0, i64 %[[N]]){{$}}
;
; CHECK-NEXT: call coldcc noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 0, i64 noundef %[[N]])
; CHECK-SAME: #[[ATTR_ALLOC:[0-9]+]]
;
; CHECK-NEXT: notail call noalias ptr addrspace(67) @llvm.kit.mobile.alloc
; CHECK-SAME: !custom-alloc ![[MD_ALLOC:[0-9]+]]
;
define void @allocate(i64 %n) {
  %1 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 0, i64 %n)
  %2 = call coldcc noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 0, i64 noundef %n) #1
  %3 = notail call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 0, i64 %n), !custom-alloc !0
  ret void
}

; CHECK-LABEL: @deallocate
; CHECK-SAME: ptr addrspace(67) %[[P:[^)]+]]
;
; CHECK-NEXT: call void @llvm.kit.mobile.free(i32 0, ptr addrspace(67) %[[P]]){{$}}
; CHECK-NEXT: call coldcc void @llvm.kit.mobile.free(i32 0, ptr addrspace(67) nonnull %[[P]])
; CHECK-SAME: #[[ATTR_FREE:[0-9]+]]
; CHECK-NEXT: notail call void @llvm.kit.mobile.free(i32 0, ptr addrspace(67) %[[P]])
; CHECK-SAME: !custom-free ![[MD_FREE:[0-9]+]]
;
define void @deallocate(ptr addrspace(67) %p) {
  call void @llvm.kit.mobile.free(i32 0, ptr addrspace(67) %p)
  call coldcc void @llvm.kit.mobile.free(i32 0, ptr addrspace(67) nonnull %p) #2
  notail call void @llvm.kit.mobile.free(i32 0, ptr addrspace(67) %p), !custom-free !1
  ret void
}

; CHECK-LABEL: @init
; CHECK-SAME: ptr addrspace(67) %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
;
; CHECK-NEXT: call void (i32, ptr addrspace(67), i64, i32, ...) @llvm.kit.mobile.init.i32(i32 0, ptr addrspace(67) %[[BUF]], i64 %[[N]], i32 1){{$}}
; CHECK-NEXT: call coldcc void (i32, ptr addrspace(67), i64, i32, ...) @llvm.kit.mobile.init.i32(i32 0, ptr addrspace(67) %[[BUF]], i64 %[[N]], i32 1)
; CHECK-SAME: #[[ATTR_INIT:[0-9]+]]
; CHECK-NEXT: notail call void (i32, ptr addrspace(67), i64, i32, ...) @llvm.kit.mobile.init.i32(i32 0, ptr addrspace(67) %[[BUF]], i64 %[[N]], i32 1)
; CHECK-SAME: !custom-init ![[MD_INIT:[0-9]+]]
;
define void @init(ptr addrspace(67) %buf, i64 %n) {
  call void(i32, ptr addrspace(67), i64, i32, ...) @llvm.kit.mobile.init(i32 0, ptr addrspace(67) %buf, i64 %n, i32 1)
  call coldcc void(i32, ptr addrspace(67), i64, i32, ...) @llvm.kit.mobile.init(i32 0, ptr addrspace(67) %buf, i64 %n, i32 1) #3
  notail call void(i32, ptr addrspace(67), i64, i32, ...) @llvm.kit.mobile.init(i32 0, ptr addrspace(67) %buf, i64 %n, i32 1), !custom-init !2
  ret void
}

; CHECK-DAG: attributes #[[ATTR_ALLOC]] = { "attr-alloc" }
; CHECK-DAG: attributes #[[ATTR_FREE]] = { "attr-free" }
; CHECK-DAG: attributes #[[ATTR_INIT]] = { "attr-init" }
attributes #1 = { "attr-alloc" }
attributes #2 = { "attr-free" }
attributes #3 = { "attr-init" }

!0 = !{!"md-alloc"}
!1 = !{!"md-free"}
!2 = !{!"md-init"}
