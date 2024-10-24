; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu --tail-dup-cbz=0 | FileCheck %s

; Function Attrs: nounwind uwtable
define void @tail_dup_merge_loops(i32 %a, ptr %b, ptr %c) local_unnamed_addr #0 {
; CHECK-LABEL: tail_dup_merge_loops:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    jmp .LBB0_1
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  .LBB0_3: # %inner_loop_exit
; CHECK-NEXT:    # in Loop: Header=BB0_1 Depth=1
; CHECK-NEXT:    incq %rsi
; CHECK-NEXT:  .LBB0_1: # %outer_loop_top
; CHECK-NEXT:    # =>This Loop Header: Depth=1
; CHECK-NEXT:    # Child Loop BB0_4 Depth 2
; CHECK-NEXT:    testl %edi, %edi
; CHECK-NEXT:    je .LBB0_5
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  # %bb.2: # %inner_loop_top
; CHECK-NEXT:    # in Loop: Header=BB0_1 Depth=1
; CHECK-NEXT:    cmpb $0, (%rsi)
; CHECK-NEXT:    js .LBB0_3
; CHECK-NEXT:  .LBB0_4: # %inner_loop_latch
; CHECK-NEXT:    # Parent Loop BB0_1 Depth=1
; CHECK-NEXT:    # => This Inner Loop Header: Depth=2
; CHECK-NEXT:    addq $2, %rsi
; CHECK-NEXT:    cmpb $0, (%rsi)
; CHECK-NEXT:    jns .LBB0_4
; CHECK-NEXT:    jmp .LBB0_3
; CHECK-NEXT:  .LBB0_5: # %exit
; CHECK-NEXT:    retq
entry:
  %notlhs674.i = icmp eq i32 %a, 0
  br label %outer_loop_top

outer_loop_top:                         ; preds = %inner_loop_exit, %entry
  %dst.0.ph.i = phi ptr [ %b, %entry ], [ %scevgep679.i, %inner_loop_exit ]
  br i1 %notlhs674.i, label %exit, label %inner_loop_preheader

inner_loop_preheader:                           ; preds = %outer_loop_top
  br label %inner_loop_top

inner_loop_top:                                     ; preds = %inner_loop_latch, %inner_loop_preheader
  %dst.0.i = phi ptr [ %inc, %inner_loop_latch ], [ %dst.0.ph.i, %inner_loop_preheader ]
  %var = load i8, ptr %dst.0.i
  %tobool1.i = icmp slt i8 %var, 0
  br label %inner_loop_test

inner_loop_test:                                       ; preds = %inner_loop_top
  br i1 %tobool1.i, label %inner_loop_exit, label %inner_loop_latch

inner_loop_exit:                       ; preds = %inner_loop_test
  %scevgep.i = getelementptr i8, ptr %dst.0.i, i64 1
  %scevgep679.i = getelementptr i8, ptr %scevgep.i, i64 0
  br label %outer_loop_top

inner_loop_latch:                                ; preds = %inner_loop_test
  %cmp75.i = icmp ult ptr %dst.0.i, %c
  %inc = getelementptr i8, ptr %dst.0.i, i64 2
  br label %inner_loop_top

exit:                              ; preds = %outer_loop_top
  ret void
}

@.str.6 = external unnamed_addr constant [23 x i8], align 1

; There is an erroneus check in LoopBase::addBasicBlockToLoop(), where it
; assumes that the header block for a loop is unique.
; For most of compilation this assumption is true, but during layout we allow
; this assumption to be violated. The following code will trigger the bug:

; The loops in question is eventually headed by the block shared_loop_header
;
; During layout The block labeled outer_loop_header gets tail-duplicated into
; outer_loop_latch, and into shared_preheader, and then removed. This leaves
; shared_loop_header as the header of both loops. The end result
; is that there are 2 valid loops, and that they share a header. If we re-ran
; the loop analysis, it would classify this as a single loop.
; So far this is fine as far as layout is concerned.
; After layout we tail merge blocks merge_other and merge_predecessor_split.
; We do this even though they share only a single instruction, because
; merge_predecessor_split falls through to their shared successor:
; outer_loop_latch.
; The rest of the blocks in the function are noise unfortunately. Bugpoint
; couldn't shrink the test any further.

define i32 @loop_shared_header(ptr %exe, i32 %exesz, i32 %headsize, i32 %min, i32 %wwprva, i32 %e_lfanew, ptr readonly %wwp, i32 %wwpsz, i16 zeroext %sects) local_unnamed_addr #0 {
; CHECK-LABEL: loop_shared_header:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pushq %rbp
; CHECK-NEXT:    pushq %r15
; CHECK-NEXT:    pushq %r14
; CHECK-NEXT:    pushq %r12
; CHECK-NEXT:    pushq %rbx
; CHECK-NEXT:    movl $1, %ebx
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    testb %al, %al
; CHECK-NEXT:    jne .LBB1_12
; CHECK-NEXT:  # %bb.1: # %if.end19
; CHECK-NEXT:    movl (%rax), %r12d
; CHECK-NEXT:    leal (,%r12,4), %ebp
; CHECK-NEXT:    movl %ebp, %r15d
; CHECK-NEXT:    movl $1, %esi
; CHECK-NEXT:    movq %r15, %rdi
; CHECK-NEXT:    callq cli_calloc@PLT
; CHECK-NEXT:    movq %rax, %r14
; CHECK-NEXT:    movb $1, %al
; CHECK-NEXT:    testb %al, %al
; CHECK-NEXT:    jne .LBB1_12
; CHECK-NEXT:  # %bb.2: # %if.end50
; CHECK-NEXT:    movq %r14, %rdi
; CHECK-NEXT:    movq %r15, %rdx
; CHECK-NEXT:    callq memcpy@PLT
; CHECK-NEXT:    cmpl $4, %ebp
; CHECK-NEXT:    jb .LBB1_19
; CHECK-NEXT:  # %bb.3: # %shared_preheader
; CHECK-NEXT:    movb $32, %cl
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    jmp .LBB1_4
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  .LBB1_15: # %merge_predecessor_split
; CHECK-NEXT:    # in Loop: Header=BB1_4 Depth=1
; CHECK-NEXT:    movb $32, %cl
; CHECK-NEXT:  .LBB1_4: # %outer_loop_header
; CHECK-NEXT:    # =>This Loop Header: Depth=1
; CHECK-NEXT:    # Child Loop BB1_8 Depth 2
; CHECK-NEXT:    testl %r12d, %r12d
; CHECK-NEXT:    je .LBB1_5
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  .LBB1_8: # %shared_loop_header
; CHECK-NEXT:    # Parent Loop BB1_4 Depth=1
; CHECK-NEXT:    # => This Inner Loop Header: Depth=2
; CHECK-NEXT:    testq %r14, %r14
; CHECK-NEXT:    jne .LBB1_18
; CHECK-NEXT:  # %bb.9: # %inner_loop_body
; CHECK-NEXT:    # in Loop: Header=BB1_8 Depth=2
; CHECK-NEXT:    testb %al, %al
; CHECK-NEXT:    je .LBB1_8
; CHECK-NEXT:  # %bb.10: # %if.end96.i
; CHECK-NEXT:    # in Loop: Header=BB1_4 Depth=1
; CHECK-NEXT:    cmpl $3, %r12d
; CHECK-NEXT:    jae .LBB1_11
; CHECK-NEXT:  # %bb.13: # %if.end287.i
; CHECK-NEXT:    # in Loop: Header=BB1_4 Depth=1
; CHECK-NEXT:    testb %al, %al
; CHECK-NEXT:    # implicit-def: $cl
; CHECK-NEXT:    jne .LBB1_4
; CHECK-NEXT:  # %bb.14: # %if.end308.i
; CHECK-NEXT:    # in Loop: Header=BB1_4 Depth=1
; CHECK-NEXT:    testb %al, %al
; CHECK-NEXT:    je .LBB1_15
; CHECK-NEXT:  # %bb.16: # %if.end335.i
; CHECK-NEXT:    # in Loop: Header=BB1_4 Depth=1
; CHECK-NEXT:    xorl %ecx, %ecx
; CHECK-NEXT:    testb %cl, %cl
; CHECK-NEXT:    jne .LBB1_4
; CHECK-NEXT:  # %bb.17: # %merge_other
; CHECK-NEXT:    # in Loop: Header=BB1_4 Depth=1
; CHECK-NEXT:    # implicit-def: $cl
; CHECK-NEXT:    jmp .LBB1_4
; CHECK-NEXT:  .LBB1_5: # %while.cond.us1412.i
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    testb %al, %al
; CHECK-NEXT:    jne .LBB1_7
; CHECK-NEXT:  # %bb.6: # %while.cond.us1412.i
; CHECK-NEXT:    decb %cl
; CHECK-NEXT:    jne .LBB1_12
; CHECK-NEXT:  .LBB1_7: # %if.end41.us1436.i
; CHECK-NEXT:  .LBB1_11: # %if.then99.i
; CHECK-NEXT:    movq .str.6@GOTPCREL(%rip), %rdi
; CHECK-NEXT:    xorl %ebx, %ebx
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    callq cli_dbgmsg@PLT
; CHECK-NEXT:  .LBB1_12: # %cleanup
; CHECK-NEXT:    movl %ebx, %eax
; CHECK-NEXT:    popq %rbx
; CHECK-NEXT:    popq %r12
; CHECK-NEXT:    popq %r14
; CHECK-NEXT:    popq %r15
; CHECK-NEXT:    popq %rbp
; CHECK-NEXT:    retq
; CHECK-NEXT:  .LBB1_18: # %wunpsect.exit.thread.loopexit389
; CHECK-NEXT:  .LBB1_19: # %wunpsect.exit.thread.loopexit391
entry:
  %0 = load i32, ptr undef, align 4
  %mul = shl nsw i32 %0, 2
  br i1 undef, label %if.end19, label %cleanup

if.end19:                                         ; preds = %entry
  %conv = zext i32 %mul to i64
  %call = tail call ptr @cli_calloc(i64 %conv, i64 1)
  %1 = icmp eq i32 %exesz, 0
  %notrhs = icmp eq i32 %0, 0
  %or.cond117.not = or i1 %1, %notrhs
  %or.cond202 = or i1 %or.cond117.not, undef
  %cmp35 = icmp ult ptr %call, %exe
  %or.cond203 = or i1 %or.cond202, %cmp35
  br i1 %or.cond203, label %cleanup, label %if.end50

if.end50:                                         ; preds = %if.end19
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull %call, ptr undef, i64 %conv, i1 false)
  %cmp1.i.i = icmp ugt i32 %mul, 3
  br i1 %cmp1.i.i, label %shared_preheader, label %wunpsect.exit.thread.loopexit391

shared_preheader:                                 ; preds = %if.end50
  br label %outer_loop_header

outer_loop_header:                                ; preds = %outer_loop_latch, %shared_preheader
  %bits.1.i = phi i8 [ 32, %shared_preheader ], [ %bits.43.i, %outer_loop_latch ]
  %dst.0.ph.i = phi ptr [ undef, %shared_preheader ], [ %scevgep679.i, %outer_loop_latch ]
  %2 = icmp eq i32 %0, 0
  br i1 %2, label %while.cond.us1412.i, label %shared_loop_header

while.cond.us1412.i:                              ; preds = %outer_loop_header
  %.pre.i = add i8 %bits.1.i, -1
  %tobool2.us1420.i = icmp eq i8 %.pre.i, 0
  %or.cond.us1421.i = or i1 undef, %tobool2.us1420.i
  br i1 %or.cond.us1421.i, label %if.end41.us1436.i, label %cleanup

if.end41.us1436.i:                                ; preds = %while.cond.us1412.i
  unreachable

shared_loop_header:                               ; preds = %dup_early2, %dup_early1
  %dst.0.i = phi ptr [ undef, %inner_loop_body ], [ %dst.0.ph.i, %outer_loop_header ], [ undef, %dead_block ]
  %cmp3.i1172.i = icmp ult ptr null, %call
  br i1 %cmp3.i1172.i, label %wunpsect.exit.thread.loopexit389, label %inner_loop_body

inner_loop_body:                                  ; preds = %shared_loop_header
  %3 = icmp slt i32 undef, 0
  br i1 %3, label %if.end96.i, label %shared_loop_header

dead_block:                                       ; preds = %inner_loop_body
  %cmp75.i = icmp ult ptr %dst.0.i, null
  br label %shared_loop_header

if.end96.i:                                       ; preds = %inner_loop_body
  %cmp97.i = icmp ugt i32 %0, 2
  br i1 %cmp97.i, label %if.then99.i, label %if.end287.i

if.then99.i:                                      ; preds = %if.end96.i
  tail call void (ptr, ...) @cli_dbgmsg(ptr @.str.6, i32 undef)
  br label %cleanup

if.end287.i:                                      ; preds = %if.end96.i
  %cmp291.i = icmp ne i32 %0, 1
  %conv294.i = select i1 %cmp291.i, i16 4, i16 3
  br i1 undef, label %if.end308.i, label %outer_loop_latch

if.end308.i:                                      ; preds = %if.end287.i
  br i1 undef, label %if.end335.i, label %merge_predecessor_split

merge_predecessor_split:                          ; preds = %if.end308.i
  %4 = bitcast ptr undef to ptr
  br label %outer_loop_latch

if.end335.i:                                      ; preds = %if.end308.i
  br i1 undef, label %outer_loop_latch, label %merge_other

merge_other:                                      ; preds = %if.end335.i
  br label %outer_loop_latch

outer_loop_latch:                                 ; preds = %merge_other, %if.end335.i, %merge_predecessor_split, %if.end287.i
  %bits.43.i = phi i8 [ undef, %if.end287.i ], [ undef, %merge_other ], [ 32, %merge_predecessor_split ], [ 0, %if.end335.i ]
  %backsize.0.i = phi i16 [ %conv294.i, %if.end287.i ], [ 0, %merge_other ], [ 0, %merge_predecessor_split ], [ 0, %if.end335.i ]
  %5 = add i16 %backsize.0.i, -1
  %6 = zext i16 %5 to i64
  %scevgep.i = getelementptr i8, ptr %dst.0.ph.i, i64 1
  %scevgep679.i = getelementptr i8, ptr %scevgep.i, i64 %6
  br label %outer_loop_header

wunpsect.exit.thread.loopexit389:                 ; preds = %shared_loop_header
  unreachable

wunpsect.exit.thread.loopexit391:                 ; preds = %if.end50
  unreachable

cleanup:                                          ; preds = %if.then99.i, %while.cond.us1412.i, %if.end19, %entry
  %retval.0 = phi i32 [ 0, %if.then99.i ], [ 1, %entry ], [ 1, %if.end19 ], [ 1, %while.cond.us1412.i ]
  ret i32 %retval.0
}

; Function Attrs: nounwind
declare void @cli_dbgmsg(ptr, ...) local_unnamed_addr #0

; Function Attrs: nounwind
declare ptr @cli_calloc(i64, i64) local_unnamed_addr #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #1
attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
