; REQUIRES: kitsune-cuda
;
; Check that attributes are preserved when lowering the kit.async.launch.kernel
; intrinsic. This intrinsic often has a combination of arguments with and
; without attributes and the runtime function does not have the same number of
; arguments as the intrinsic. Unlike the other runtime functions whose signature
; does not match that of the corresponding intrinsic, in this case, the
; signature of the runtime function is unlikely to change, so we use this to
; test that we handle such a complicated case correctly.
;
; ------------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @launch
; CHECK: %[[ARRAY2:[0-9]+]] = alloca [3 x ptr]
; CHECK: %[[ARRAY1:[0-9]+]] = alloca [4 x ptr]
; CHECK: call ptr @__kitcuda_launch_kernel(ptr nonnull @fb, ptr nonnull @1, ptr nonnull %[[ARRAY1]], i64 %n, i32 0, ptr nonnull @0, ptr %p)
; CHECK: call ptr @__kitcuda_launch_kernel(ptr nonnull @fb, ptr nonnull @1, ptr nonnull %[[ARRAY2]], i64 %n, i32 0, ptr nonnull @0, ptr %p) #[[ATTRS:[0-9]+]]
; CHECK: ret void
;
; CHECK: attributes #[[ATTRS]] = { "custom-attr" }

target triple = "x86_64-unknown-linux-gnu"

@fb = external global [23 x i8]
@0 = external global i32
@1 = external global float

define void @launch(ptr %p, ptr nonnull %q, i64 %n, float %f) {
  call ptr (i32, ptr, ptr, i64, i32, ptr, ptr, ...) @llvm.kit.async.launch.kernel(i32 2, ptr nonnull @fb, ptr nonnull @1, i64 %n, i32 0, ptr nonnull @0, ptr %p, ptr dereferenceable(32) %q, i32 noundef 98, float %f, ptr null)
  call ptr (i32, ptr, ptr, i64, i32, ptr, ptr, ...) @llvm.kit.async.launch.kernel(i32 2, ptr nonnull @fb, ptr nonnull @1, i64 %n, i32 0, ptr nonnull @0, ptr %p, ptr dereferenceable(32) %q, i32 noundef 98, ptr null) #0
  ret void
}

attributes #0 = { "custom-attr" }
