; Check that kernel functions with warp shuffle reductions are finalized
; correctly.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt -passes='emb-finalize-reductions' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s

; CHECK-LABEL: @f10
; CHECK-SAME: ptr %[[R32:[^,]+]]
; CHECK-SAME: ptr %[[R64:[^,]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[L32:.+]] = alloca i32
; CHECK-NEXT: %[[L64:.+]] = alloca i64
; CHECK-NEXT: store i32 -1, ptr %[[L32]]
; CHECK-NEXT: store i64 0, ptr %[[L64]]
; CHECK: br i1 %{{.+}}, label %[[BODY:.+]], label %[[EXIT:.+]]
; CHECK: [[BODY]]:
; CHECK: [[HEADER:.+]]:
; CHECK-NEXT: %[[I64:.+]] = phi i64
; CHECK-SAME: [ 0, %[[BODY]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[HEADER]] ]
; CHECK-NEXT: %[[I32:.+]] = trunc i64 %[[I64]] to i32
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce.i32
; CHECK-SAME: i32 4, i32 27, ptr %[[L32]], i32 %[[I32]], i32 1
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce.i64
; CHECK-SAME: i32 4, i32 5, ptr %[[L64]], i64 %[[I64]], i32 1
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[I64]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[FINAL:.+]], label %[[HEADER]]
; CHECK-EMPTY:
; CHECK-NEXT: [[FINAL]]:
; CHECK-NEXT: %[[UNUSED_I32:.+]] = load
; CHECK-NEXT: %[[UNUSED_I64:.+]] = load
; CHECK-NEXT: br label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: call void @llvm.kit.gpu.warp.sync(i32 4)
; CHECK-NEXT: %[[V32:.+]] = load i32, ptr %[[L32]]
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce.i32
; CHECK-SAME: i32 4, i32 27, ptr %[[R32]], i32 %[[V32]], i32 10
; CHECK-NEXT: call void @llvm.kit.gpu.warp.sync(i32 4)
; CHECK-NEXT: %[[V64:.+]] = load i64, ptr %[[L64]]
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce.i64
; CHECK-SAME: i32 4, i32 5, ptr %[[R64]], i64 %[[V64]], i32 10
; CHECK-NEXT: ret void
define void @f10(ptr %r32, ptr %r64, i64 %tc) !kit.func !0 {
entry:
  %l32 = alloca i32
  %l64 = alloca i64
  %idx = call i64 @index()
  %end = add i64 %idx, 1
  %cmpidx = icmp ult i64 %idx, %tc
  br i1 %cmpidx, label %body, label %exit

body:
  store i32 -1, ptr %l32
  store i64 0, ptr %l64
  br label %header

header:
  %i64 = phi i64 [ 0, %body ], [ %inc, %header ]
  %i32 = trunc i64 %i64 to i32
  call void (i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce.i32(i32 4, i32 27, ptr %l32, i32 %i32, i32 1, i32 4, i32 -1, ptr @umin)
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce.i64(i32 4, i32 5, ptr %l64, i64 %i64, i32 1, i32 8, i64 0, ptr @sum)
  %inc = add i64 %i64, 1
  %cmp = icmp eq i64 %inc, %end
  br i1 %cmp, label %final, label %header, !llvm.loop !0

final:
  %v32 = load i32, ptr %l32
  %v64 = load i64, ptr %l64
  call void (i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce.i32(i32 4, i32 27, ptr %r32, i32 %v32, i32 10, i32 4, i32 -1, ptr @umin)
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce.i64(i32 4, i32 5, ptr %r64, i64 %v64, i32 10, i32 8, i64 0, ptr @sum)
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @f14
; CHECK-SAME: ptr %[[R32:[^,]+]]
; CHECK-SAME: ptr %[[R64:[^,]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[L32:.+]] = alloca i32
; CHECK-NEXT: %[[L64:.+]] = alloca i64
; CHECK-NEXT: store i32 -1, ptr %[[L32]]
; CHECK-NEXT: store i64 0, ptr %[[L64]]
; CHECK: br i1 %{{.+}}, label %[[BODY:.+]], label %[[EXIT:.+]]
; CHECK: [[BODY]]:
; CHECK: [[HEADER:.+]]:
; CHECK-NEXT: %[[I64:.+]] = phi i64
; CHECK-SAME: [ 0, %[[BODY]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[HEADER]] ]
; CHECK-NEXT: %[[I32:.+]] = trunc i64 %[[I64]] to i32
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce.i32
; CHECK-SAME: i32 4, i32 27, ptr %[[L32]], i32 %[[I32]], i32 1
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce.i64
; CHECK-SAME: i32 4, i32 5, ptr %[[L64]], i64 %[[I64]], i32 1
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[I64]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[FINAL:.+]], label %[[HEADER]]
; CHECK-EMPTY:
; CHECK-NEXT: [[FINAL]]:
; CHECK-NEXT: %[[UNUSED_I32:.+]] = load
; CHECK-NEXT: %[[UNUSED_I64:.+]] = load
; CHECK-NEXT: br label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: call void @llvm.kit.gpu.warp.sync(i32 4)
; CHECK-NEXT: %[[V32:.+]] = load i32, ptr %[[L32]]
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce.i32
; CHECK-SAME: i32 4, i32 27, ptr %[[R32]], i32 %[[V32]], i32 14
; CHECK-NEXT: call void @llvm.kit.gpu.warp.sync(i32 4)
; CHECK-NEXT: %[[V64:.+]] = load i64, ptr %[[L64]]
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce.i64
; CHECK-SAME: i32 4, i32 5, ptr %[[R64]], i64 %[[V64]], i32 14
; CHECK-NEXT: ret void
define void @f14(ptr %r32, ptr %r64, i64 %tc) !kit.func !2 {
entry:
  %l32 = alloca i32
  %l64 = alloca i64
  %idx = call i64 @index()
  %end = add i64 %idx, 1
  %cmpidx = icmp ult i64 %idx, %tc
  br i1 %cmpidx, label %body, label %exit

body:
  store i32 -1, ptr %l32
  store i64 0, ptr %l64
  br label %header

header:
  %i64 = phi i64 [ 0, %body ], [ %inc, %header ]
  %i32 = trunc i64 %i64 to i32
  call void (i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce.i32(i32 4, i32 27, ptr %l32, i32 %i32, i32 1, i32 4, i32 -1, ptr @umin)
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce.i64(i32 4, i32 5, ptr %l64, i64 %i64, i32 1, i32 8, i64 0, ptr @sum)
  %inc = add i64 %i64, 1
  %cmp = icmp eq i64 %inc, %end
  br i1 %cmp, label %final, label %header, !llvm.loop !3

final:
  %v32 = load i32, ptr %l32
  %v64 = load i64, ptr %l64
  call void (i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce.i32(i32 4, i32 27, ptr %r32, i32 %v32, i32 14, i32 4, i32 -1, ptr @umin)
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce.i64(i32 4, i32 5, ptr %r64, i64 %v64, i32 14, i32 8, i64 0, ptr @sum)
  br label %exit

exit:
  ret void
}

declare i64 @index()
declare void @umin(ptr, i32)
declare void @sum(ptr, i64)

!0 = distinct !{!0, !4}
!1 = distinct !{!1}
!2 = distinct !{!2, !4}
!3 = distinct !{!3}
!4 = !{!"kit.func.kernel", i32 1}
