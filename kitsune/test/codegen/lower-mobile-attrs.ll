; Check that all attributes are preserved when lowering Kitsune's mobile
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
; CHECK-SAME: #[[ATTR_ALLOC:[0-9]+]]
; CHECK: call noalias ptr @__kitrt_default_mem_alloc(i64 noundef %[[N]]){{$}}
; CHECK: call noalias ptr @__kitrt_default_mem_alloc(i64 noundef %[[N]])
; CHECK-SAME: #[[ATTR_ALLOC]]
define void @allocate(i64 %n) {
  %1 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %n)
  %2 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %n) #1
  %3 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 noundef %n)
  %4 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 noundef %n) #1
  ret void
}

; CHECK-LABEL: @deallocate
; CHECK-SAME: ptr addrspace(67) %[[P:[^)]+]]
; CHECK: call void @__kitrt_default_mem_free(ptr %{{.+}}){{$}}
; CHECK: call void @__kitrt_default_mem_free(ptr %{{.+}})
; CHECK-SAME: #[[ATTR_FREE:[0-9]+]]
; CHECK: call void @__kitrt_default_mem_free(ptr nonnull %{{.+}}){{$}}
; CHECK: call void @__kitrt_default_mem_free(ptr nonnull %{{.+}})
; CHECK-SAME: #[[ATTR_FREE]]
define void @deallocate(ptr addrspace(67) %p) {
  call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %p)
  call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %p) #2
  call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) nonnull %p)
  call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) nonnull %p) #2
  ret void
}

; CHECK-LABEL: @init
; CHECK-SAME: ptr addrspace(67) %[[BUF:[^,]+]]
; CHECK-SAME: i64 noundef %[[M:[^,]+]]
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: float %[[V:[^)]+]]
; CHECK: call void @__kitrt_mobile_init_float(ptr %{{.+}}, i64 %[[N]], float %[[V]]){{$}}
; CHECK: call void @__kitrt_mobile_init_float(ptr %{{.+}}, i64 %[[N]], float %[[V]])
; CHECK-SAME: #[[ATTR_INIT:[0-9]+]]
; CHECK: call void @__kitrt_mobile_init_float(ptr nonnull %{{.+}}, i64 noundef %[[M]], float %[[V]]){{$}}
; CHECK: call void @__kitrt_mobile_init_float(ptr nonnull %{{.+}}, i64 noundef %[[M]], float %[[V]])
; CHECK-SAME: #[[ATTR_INIT]]
define void @init(ptr addrspace(67) %buf, i64 noundef %m, i64 %n, float %v) {
  call void (i32, ptr addrspace(67), i64, float, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, float %v)
  call void (i32, ptr addrspace(67), i64, float, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, float %v) #3
  call void (i32, ptr addrspace(67), i64, float, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) nonnull %buf, i64 noundef %m, float %v)
  call void (i32, ptr addrspace(67), i64, float, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) nonnull %buf, i64 noundef %m, float %v) #3
  ret void
}

; CHECK-DAG: attributes #[[ATTR_ALLOC]] = { "attr-alloc" }
; CHECK-DAG: attributes #[[ATTR_FREE]] = { "attr-free" }
; CHECK-DAG: attributes #[[ATTR_INIT]] = { "attr-init" }
attributes #1 = { "attr-alloc" }
attributes #2 = { "attr-free" }
attributes #3 = { "attr-init" }
