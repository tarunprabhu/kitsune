; Calls to the intrinsic that are in different basic blocks are not grouped,
; even if there is an unconditional branch between the blocks, and the two
; blocks could, in principle, be merged.
;
; RUN: opt -passes='kit-lower-reduce-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f1
; CHECK-SAME: ptr {{[^%]*}}%[[BUF1:[^,]+]]
; CHECK-SAME: ptr {{[^%]*}}%[[BUF2:[^,]+]]
; CHECK-SAME: ptr {{[^%]*}}%[[BUF3:[^,]+]]
; CHECK-SAME: ptr {{[^%]*}}%[[BUF4:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[R1:.+]] = alloca i32
; CHECK-NEXT: %[[R2:.+]] = alloca i32
; CHECK-NEXT: %[[R3:.+]] = alloca i32
; CHECK-NEXT: %[[R4:.+]] = alloca i32
; CHECK-NEXT: br label %[[PH1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH1]]:
; CHECK-NEXT: br label %[[HEADER1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER1]]:
; CHECK-NEXT: %[[IV1:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH1]] ]
; CHECK-SAME: [ %[[INC1:.+]], %[[HEADER1]] ]
; CHECK-NEXT: %[[ADDR1:.+]] = getelementptr {{.+}} %[[BUF1]], i64 %[[IV1]]
; CHECK-NEXT: %[[V1:.+]] = load i32, ptr {{.*}}%[[ADDR1]]
; CHECK-NEXT: call void @mul(ptr %[[R1]], i32 %[[V1]])
; CHECK-NEXT: %[[ADDR2:.+]] = getelementptr {{.+}} %[[BUF2]], i64 %[[IV1]]
; CHECK-NEXT: %[[V2:.+]] = load i32, ptr {{.*}}%[[ADDR2]]
; CHECK-NEXT: call void @mul(ptr %[[R2]], i32 %[[V2]])
; CHECK-NEXT: %[[INC1:.+]] = add i64 %[[IV1]], 1
; CHECK-NEXT: %[[CMP1:.+]] = icmp eq i64 %[[INC1]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP1]], label %[[EXIT1:.+]], label %[[HEADER1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT1]]:
; CHECK-NEXT: br label %[[PH2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH2]]:
; CHECK-NEXT: br label %[[HEADER2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER2]]:
; CHECK-NEXT: %[[IV2:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH2]] ]
; CHECK-SAME: [ %[[INC2:.+]], %[[HEADER2]] ]
; CHECK-NEXT: %[[ADDR3:.+]] = getelementptr {{.+}} %[[BUF3]], i64 %[[IV2]]
; CHECK-NEXT: %[[V3:.+]] = load i32, ptr {{.*}}%[[ADDR3]]
; CHECK-NEXT: call void @mul(ptr %[[R3]], i32 %[[V3]])
; CHECK-NEXT: %[[ADDR4:.+]] = getelementptr {{.+}} %[[BUF4]], i64 %[[IV2]]
; CHECK-NEXT: %[[V4:.+]] = load i32, ptr {{.*}}%[[ADDR4]]
; CHECK-NEXT: call void @mul(ptr %[[R4]], i32 %[[V4]])
; CHECK-NEXT: %[[INC2:.+]] = add i64 %[[IV2]], 1
; CHECK-NEXT: %[[CMP2:.+]] = icmp eq i64 %[[INC2]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP2]], label %[[EXIT2:.+]], label %[[HEADER2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT2]]:
; CHECK-NEXT: br label %[[END:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END]]:
; CHECK-NEXT: ret void

declare void @mul(ptr %res, i32 %v)

define void @f1(ptr addrspace(67) %buf1, ptr addrspace(67) %buf2, ptr addrspace(67) %buf3, ptr addrspace(67) %buf4, i64 %n) {
entry:
  %r1 = alloca i32
  %r2 = alloca i32
  %r3 = alloca i32
  %r4 = alloca i32
  br label %group1

group1:
  call void(i32, ptr, i32, ptr addrspace(67) , i64, i32, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %r1, i32 4, ptr addrspace(67) %buf1, i64 %n, i32 1, ptr @mul)
  call void(i32, ptr, i32, ptr addrspace(67) , i64, i32, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %r2, i32 4, ptr addrspace(67) %buf2, i64 %n, i32 1, ptr @mul)
  br label %group2

group2:
  call void(i32, ptr, i32, ptr addrspace(67) , i64, i32, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %r3, i32 4, ptr addrspace(67) %buf3, i64 %n, i32 1, ptr @mul)
  call void(i32, ptr, i32, ptr addrspace(67) , i64, i32, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %r4, i32 4, ptr addrspace(67) %buf4, i64 %n, i32 1, ptr @mul)
  br label %end

end:
  ret void
}
