define void @pp(ptr %a, ptr %b, i64 %m, i64 %n) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  %0 = mul i64 %n, %i
  %1 = add i64 %0, %j
  %2 = srem i64 %i, 2
  %3 = icmp eq i64 %2, 0
  br i1 %3, label %even, label %odd

even:
  %aidx = getelementptr i64, ptr %a, i64 %i
  store i64 %1, ptr %aidx
  br label %callext

odd:
  %bidx = getelementptr i64, ptr %b, i64 %i
  store i64 %1, ptr %bidx
  br label %callext

callext:
  call void @ext(i64 %1)
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 2}
!1 = distinct !{!1, !0, !3}
!2 = distinct !{!2, !0, !3}
!3 = !{!"tapir.loop.spawn.strategy", i32 1}

declare void @ext(i64)
