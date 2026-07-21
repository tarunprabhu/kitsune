; Check that instruction metadata is preserved when lowering Kitsune's mobile
; intrinsics. The handling is the same for all tapir targets, so checking this
; with the 'serial' tapir target is sufficient.
;
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s

target triple = "x86_64-pc-linux-gnu"

; CHECK-LABEL: @allocate
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK: call noalias ptr @__kitrt_default_mem_alloc(i64 %[[N]]){{$}}
; CHECK: call noalias ptr @__kitrt_default_mem_alloc(i64 %[[N]])
; CHECK-SAME: !custom-alloc ![[MD_ALLOC:[0-9]+]]
define void @allocate(i64 %n) {
  %1 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %n)
  %2 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %n), !custom-alloc !0
  ret void
}

; CHECK-LABEL: @deallocate
; CHECK-SAME: ptr addrspace(67) %[[P:[^)]+]]
; CHECK: call void @__kitrt_default_mem_free(ptr %{{.+}}){{$}}
; CHECK: call void @__kitrt_default_mem_free(ptr %{{.+}})
; CHECK-SAME: !custom-free ![[MD_FREE:[0-9]+]]
define void @deallocate(ptr addrspace(67) %p) {
  call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %p)
  call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %p), !custom-free !1
  ret void
}

; CHECK-LABEL: @init
; CHECK-SAME: ptr addrspace(67) %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: i16 %[[V:[^)]+]]
; CHECK: call void @__kitrt_mobile_init_i16(ptr %{{[^,]+}}, i64 %[[N]], i16 %[[V]]){{$}}
; CHECK: call void @__kitrt_mobile_init_i16(ptr %{{[^,]+}}, i64 %[[N]], i16 %[[V]])
; CHECK-SAME: !custom-init ![[MD_INIT:[0-9]+]]
define void @init(ptr addrspace(67) %buf, i64 %n, i16 %v) {
  call void (i32, ptr addrspace(67), i64, i16, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, i16 %v)
  call void (i32, ptr addrspace(67), i64, i16, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, i16 %v), !custom-init !2
  ret void
}

; CHECK-DAG: ![[MD_ALLOC]] = !{!"md-alloc"}
; CHECK-DAG: ![[MD_FREE]] = !{!"md-free"}
; CHECK-DAG: ![[MD_INIT]] = !{!"md-init"}
!0 = !{!"md-alloc"}
!1 = !{!"md-free"}
!2 = !{!"md-init"}
