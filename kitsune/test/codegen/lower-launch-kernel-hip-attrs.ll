; REQUIRES: kitsune-hip
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
; RUN: opt --tapir=hip -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s --check-prefix HIP
;
; HIP-LABEL: @launch
; HIP-NEXT: call ptr @__kithip_launch_kernel(ptr nonnull @fb, ptr nonnull @1, ptr nonnull %q, i64 %n, i32 0, ptr nonnull @0, ptr %p)
; HIP-NEXT: call ptr @__kithip_launch_kernel(ptr nonnull @fb, ptr nonnull @1, ptr nonnull %q, i64 %n, i32 0, ptr nonnull @0, ptr %p) #[[ATTRS:[0-9]+]]
; HIP-NEXT: ret void
;
; HIP: attributes #[[ATTRS]] = { "custom-attr" }

target triple = "x86_64-unknown-linux-gnu"

@fb = external global [23 x i8]
@0 = external global i32
@1 = external global float

define void @launch(ptr %p, ptr nonnull %q, i64 %n) {
  call ptr @llvm.kit.async.launch.kernel(i32 4, ptr nonnull @fb, ptr nonnull @1, ptr nonnull %q, i64 %n, i32 0, ptr nonnull @0, ptr %p)
  call ptr @llvm.kit.async.launch.kernel(i32 4, ptr nonnull @fb, ptr nonnull @1, ptr nonnull %q, i64 %n, i32 0, ptr nonnull @0, ptr %p) #0
  ret void
}

attributes #0 = { "custom-attr" }
