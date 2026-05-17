; Check that calls to @kit.reduce.1 intrinsic are lowered correctly.
;
; RUN: opt --passes='kit-lower-reduce-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-NOT: @llvm.kit.reduce.0

declare void @sum(ptr %res, double %v)

; CHECK-LABEL: @fsum
; CHECK-SAME: ptr {{[^%]*}}%[[ARR:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[RES:.+]] = alloca double
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[IV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[HEADER]] ]
; CHECK-NEXT: %[[ADDR:.+]] = getelementptr {{.+}} %[[ARR]], i64 %[[IV]]
; CHECK-NEXT: %[[V:.+]] = load double, ptr {{.*}}%[[ADDR]]
; CHECK-NEXT: call void @sum(ptr %[[RES]], double %[[V]])
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[IV]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
define void @fsum(ptr addrspace(67) %arr, i64 %n) {
entry:
  %res = alloca double
  call void(i32, ptr, i32, ptr addrspace(67), i64, double, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %res, i32 8, ptr addrspace(67) %arr, i64 %n, double 0.0, ptr @sum)
  ret void
}

declare void @mul(ptr %res, i32 %v)

; CHECK-LABEL: @fmul
; CHECK-SAME: ptr {{[^%]*}}%[[ARR:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[RES:.+]] = alloca i32
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[IV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[HEADER]] ]
; CHECK-NEXT: %[[ADDR:.+]] = getelementptr {{.+}} %[[ARR]], i64 %[[IV]]
; CHECK-NEXT: %[[V:.+]] = load i32, ptr {{.*}}%[[ADDR]]
; CHECK-NEXT: call void @mul(ptr %[[RES]], i32 %[[V]])
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[IV]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
define void @fmul(ptr addrspace(67) %arr, i64 %n) {
entry:
  %res = alloca i32
  call void(i32, ptr, i32, ptr addrspace(67), i64, i32, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %res, i32 4, ptr addrspace(67) %arr, i64 %n, i32 1, ptr @mul)
  ret void
}

declare void @custom(ptr %res, ptr %obj, ptr %class, i64 %more)

; CHECK-LABEL: @fcustom
; CHECK-SAME: ptr %[[RES:[^,]+]]
; CHECK-SAME: ptr {{[^%]*}}%[[ARR:[^,]+]]
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: ptr %[[CLASS:[^,]+]]
; CHECK-SAME: i64 %[[MORE:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[IV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[HEADER]] ]
; CHECK-NEXT: %[[ADDR:.+]] = getelementptr {{.+}} %[[ARR]], i64 %[[IV]]
; CHECK-NEXT: %[[V:.+]] = load ptr, ptr {{.*}}%[[ADDR]]
; CHECK-NEXT: call void @custom(ptr %[[RES]], ptr %[[V]], ptr %[[CLASS]], i64 %[[MORE]])
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[IV]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
define void @fcustom(ptr %res, ptr addrspace(67) %arr, i64 %n, ptr %class, i64 %more) {
entry:
  call void (i32, ptr, i32, ptr addrspace(67), i64, ptr, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %res, i32 64, ptr addrspace(67) %arr, i64 %n, ptr null, ptr @custom, ptr %class, i64 %more)
  ret void
}

; CHECK-LABEL: @ext
; CHECK-SAME: ptr {{[^%]*}}%[[ARR:[^,]+]]
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: ptr %[[REDUCER:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[RES:.+]] = alloca double
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[IV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[HEADER]] ]
; CHECK-NEXT: %[[ADDR:.+]] = getelementptr {{.+}} %[[ARR]], i64 %[[IV]]
; CHECK-NEXT: %[[V:.+]] = load double, ptr {{.*}}%[[ADDR]]
; CHECK-NEXT: call void %[[REDUCER]](ptr %[[RES]], double %[[V]])
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[IV]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
define void @ext(ptr addrspace(67) %arr, i64 %n, ptr %reducer) {
entry:
  %res = alloca double
  call void(i32, ptr, i32, ptr addrspace(67), i64, double, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %res, i32 8, ptr addrspace(67) %arr, i64 %n, double 0.0, ptr %reducer)
  ret void
}

; CHECK-LABEL: @ext_vararg
; CHECK-SAME: ptr {{[^%]*}}%[[ARR:[^,]+]]
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: ptr %[[REDUCER:[^,]+]]
; CHECK-SAME: ptr %[[CLASS:[^,]+]]
; CHECK-SAME: i8 %[[MORE:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[RES:.+]] = alloca float
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[IV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[HEADER]] ]
; CHECK-NEXT: %[[ADDR:.+]] = getelementptr {{.+}} %[[ARR]], i64 %[[IV]]
; CHECK-NEXT: %[[V:.+]] = load float, ptr {{.*}}%[[ADDR]]
; CHECK-NEXT: call void %[[REDUCER]](ptr %[[RES]], float %[[V]], ptr %[[CLASS]], i8 %[[MORE]])
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[IV]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
define void @ext_vararg(ptr addrspace(67) %arr, i64 %n, ptr %reducer, ptr %class, i8 %more) {
entry:
  %res = alloca float
  call void(i32, ptr, i32, ptr addrspace(67), i64, float, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %res, i32 8, ptr addrspace(67) %arr, i64 %n, float 0.0, ptr %reducer, ptr %class, i8 %more)
  ret void
}
