; Check that syncing threads launched by the pthreads tapir target is lowered
; correctly.
;
; RUN: opt --tapir=pthreads -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @launch
; CHECK: call void @__kitpthr_sync(ptr nonnull dereferenceable(16) %ctx)

target triple = "x86_64-unknown-linux-gnu"

define void @launch(ptr nonnull dereferenceable(16) %ctx) {
  call void (i32, ptr) @llvm.kit.sync.threads(i32 1024, ptr nonnull dereferenceable(16) %ctx)
  ret void
}
