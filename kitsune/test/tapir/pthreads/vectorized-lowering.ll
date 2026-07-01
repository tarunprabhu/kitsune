; Check that the default lowering of a simple tapir loop produces vectorized
; code.
;
; RUN: %if x86-registered-target %{ \
; RUN:   opt -mtriple=x86_64-pc-linux --tapir=pthreads -O3 -S %s \
; RUN:       | FileCheck %s --check-prefix=X86 \
; RUN: %}
;
; X86-LABEL: define {{.*}}void @f.outline{{[^ ]+}}.ls1(
; X86: %[[A1:.+]] = load <4 x float>
; X86: %[[A2:.+]] = load <4 x float>
; X86: %[[B1:.+]] = load <4 x float>
; X86: %[[B2:.+]] = load <4 x float>
; X86: %[[SUM1:.+]] = fadd <4 x float> %[[A1]], %[[B1]]
; X86: %[[SUM2:.+]] = fadd <4 x float> %[[A2]], %[[B2]]
; X86: store <4 x float> %[[SUM1]]
; X86: store <4 x float> %[[SUM2]]
;
;
; RUN: %if aarch64-registered-target %{ \
; RUN:   opt -mtriple=aarch64-linux-gnu -mattr=+sve --tapir=pthreads -O3 -S %s \
; RUN:       | FileCheck %s --check-prefix=AARCH64 \
; RUN: %}
;
; AARCH64-LABEL: define {{.*}}void @f.outline{{[^ ]+}}.ls1(
; AARCH64: %[[A1:.+]] = load <vscale x 4 x float>
; AARCH64: %[[A2:.+]] = load <vscale x 4 x float>
; AARCH64: %[[B1:.+]] = load <vscale x 4 x float>
; AARCH64: %[[B2:.+]] = load <vscale x 4 x float>
; AARCH64: %[[SUM1:.+]] = fadd <vscale x 4 x float> %[[A1]], %[[B1]]
; AARCH64: %[[SUM2:.+]] = fadd <vscale x 4 x float> %[[A2]], %[[B2]]
; AARCH64: store <vscale x 4 x float> %[[SUM1]]
; AARCH64: store <vscale x 4 x float> %[[SUM2]]

define void @f(ptr %a, ptr %b, ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  %aidx = getelementptr float, ptr %a, i64 %i
  %aelem = load float, ptr %aidx
  %bidx = getelementptr float, ptr %b, i64 %i
  %belem = load float, ptr %bidx
  %sum = fadd float %aelem, %belem
  %cidx = getelementptr float, ptr %c, i64 %i
  store float %sum, ptr %cidx
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg, label %for.i.end

for.i.end:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.target", i32 1024}
!2 = !{!"tapir.loop.spawn.strategy", i32 4}
