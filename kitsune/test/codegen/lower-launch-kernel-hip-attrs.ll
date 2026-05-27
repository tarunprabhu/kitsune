; REQUIRES: kitsune-hip
;
; Check that attributes are preserved when lowering the kit.async.launch.kernel
; intrinsic. This intrinsic often has a combination of arguments with and
; without attributes. The runtime function does not have the same number of
; arguments as the intrinsic. This is intended to test that the attributes are
; copied over correctly to the runtime function that is called.
;
; RUN: opt --tapir=hip -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @launch
; CHECK: %[[ARRAY2:[0-9]+]] = alloca [3 x ptr]
; CHECK: %[[ARRAY1:[0-9]+]] = alloca [4 x ptr]
; CHECK: call ptr @__kithip_launch_kernel(ptr @fb, ptr @1, ptr nonnull %[[ARRAY1]], i64 %n, i64 0, i64 -1, i32 0, ptr @0, ptr %p){{$}}
; CHECK: %[[CTX:[0-9]+]] = call ptr @__kithip_launch_kernel(ptr nonnull @fb, ptr nonnull @1, ptr nonnull %[[ARRAY2]], i64 %n, i64 0, i64 -1, i32 0, ptr nonnull @0, ptr %p) #[[LAUNCH:[0-9]+]]
; CHECK: call void @__kithip_sync_thread_stream(ptr %[[CTX]]) #[[SYNC:[0-9]+]]
; CHECK: ret void
;
; CHECK: attributes #[[LAUNCH]] = { "launch" }
; CHECK: attributes #[[SYNC]] = { "sync" }

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@fb = external global [23 x i8]
@0 = external global i32
@1 = external global float

define void @launch(ptr %p, ptr nonnull %q, i64 %n, float %f) {
  %1 = call ptr (i32, ptr, ptr, i64, i64, i64, i32, ptr, ptr, ...) @llvm.kit.async.gpu.kernel.launch(i32 4, ptr @fb, ptr @1, i64 %n, i64 0, i64 -1, i32 0, ptr @0, ptr %p, ptr %q, i32 98, float %f, ptr null)
  %2 = call ptr (i32, ptr, ptr, i64, i64, i64, i32, ptr, ptr, ...) @llvm.kit.async.gpu.kernel.launch(i32 4, ptr nonnull @fb, ptr nonnull @1, i64 %n, i64 0, i64 -1, i32 0, ptr nonnull @0, ptr %p, ptr dereferenceable(32) %q, i32 noundef 98, ptr null) #0
  call void @llvm.kit.gpu.stream.sync(i32 4, ptr %2) #1
  ret void
}

attributes #0 = { "launch" }
attributes #1 = { "sync" }
