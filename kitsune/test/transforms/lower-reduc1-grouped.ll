; Check that successive calls to kit.reduce.1 are grouped together during
; lowering. This checks several things at once.
;
;   - The sizes of the reduction results are not the same, but the calls to the
;     intrinsic should be grouped
;
;   - The unit values and their types are irrelevant, as is the actual reducer
;     function
;
;   - If additional parameters are to be passed to the reducer, they are passed
;     and this has no effect on the grouping.
;
; RUN: opt -passes='kit-lower-reduce-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f1
; CHECK-SAME: ptr {{[^%]*}}%[[BUF1:[^,]+]]
; CHECK-SAME: ptr {{[^%]*}}%[[BUF2:[^,]+]]
; CHECK-SAME: ptr {{[^%]*}}%[[BUF3:[^,]+]]
; CHECK-SAME: ptr {{[^%]*}}%[[BUF4:[^,]+]]
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: ptr %[[REDUCER:[^,]+]]
; CHECK-SAME: ptr %[[EXTRA:[^,]+]]
; CHECK-SAME: i8 %[[MORE:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[R1:.+]] = alloca i32
; CHECK-NEXT: %[[R2:.+]] = alloca i64
; CHECK-NEXT: %[[R3:.+]] = alloca float
; CHECK-NEXT: %[[R4:.+]] = alloca ptr
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[IV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[HEADER]] ]
;
; CHECK-NEXT: %[[ADDR1:.+]] = getelementptr {{.+}} %[[BUF1]], i64 %[[IV]]
; CHECK-NEXT: %[[V1:.+]] = load i32, ptr {{.*}}%[[ADDR1]]
; CHECK-NEXT: call void @sum(ptr %[[R1]], i32 %[[V1]])
;
; CHECK-NEXT: %[[ADDR2:.+]] = getelementptr {{.+}} %[[BUF2]], i64 %[[IV]]
; CHECK-NEXT: %[[V2:.+]] = load i64, ptr {{.*}}%[[ADDR2]]
; CHECK-NEXT: call void @mul(ptr %[[R2]], i64 %[[V2]])
;
; CHECK-NEXT: %[[ADDR3:.+]] = getelementptr {{.+}} %[[BUF3]], i64 %[[IV]]
; CHECK-NEXT: %[[V3:.+]] = load float, ptr {{.*}}%[[ADDR3]]
; CHECK-NEXT: call void %[[REDUCER]](ptr %[[R3]], float %[[V3]])
;
; CHECK-NEXT: %[[ADDR4:.+]] = getelementptr {{.+}} %[[BUF4]], i64 %[[IV]]
; CHECK-NEXT: %[[V4:.+]] = load ptr, ptr {{.*}}%[[ADDR4]]
; CHECK-NEXT: call void @custom(ptr %[[R4]], ptr %[[V4]], ptr %[[EXTRA]], i8 %[[MORE]])
;
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[IV]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: call void @ext()
; CHECK-NEXT: ret void

declare void @ext()
declare void @sum(ptr %res, i32 %v)
declare void @mul(ptr %res, i64 %v)
declare void @custom(ptr %res, ptr %v, ptr %extra, i8 %more)

define void @f1(ptr addrspace(67) %buf1, ptr addrspace(67) %buf2, ptr addrspace(67) %buf3, ptr addrspace(67) %buf4, i64 %n, ptr %reducer, ptr %extra, i8 %more) {
entry:
  %r1 = alloca i32
  %r2 = alloca i64
  %r3 = alloca float
  %r4 = alloca ptr
  call void(i32, ptr, i32, ptr addrspace(67) , i64, i32, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %r1, i32 4, ptr addrspace(67) %buf1, i64 %n, i32 0, ptr @sum)
  call void(i32, ptr, i32, ptr addrspace(67) , i64, i64, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %r2, i32 8, ptr addrspace(67) %buf2, i64 %n, i64 1, ptr @mul)
  call void(i32, ptr, i32, ptr addrspace(67) , i64, float, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %r3, i32 4, ptr addrspace(67) %buf3, i64 %n, float 1.0, ptr %reducer)
  call void(i32, ptr, i32, ptr addrspace(67) , i64, ptr, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %r4, i32 48, ptr addrspace(67) %buf4, i64 %n, ptr null, ptr @custom, ptr %extra, i8 %more)
  call void @ext()
  ret void
}
