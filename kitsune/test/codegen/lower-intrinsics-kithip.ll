; REQUIRES: kitsune-hip
;
; Check that intrinsics that map to Kitsune's hip runtime are lowered correctly.
; If more intrinsics are created, they should be added here to test basic
; intrinsic lowering.
;
; RUN: opt --tapir=hip -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]
@.gname = unnamed_addr constant [5 x i8] c "gbuf\00"
@.name = unnamed_addr constant [7 x i8] c"kernel\00"

; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
define void @f(ptr %buf, i64 %n) {
  ; CHECK-NEXT: %[[SLOT0:.+]] = alloca ptr
  ; CHECK-NEXT: %[[BUNDLE:.+]] = alloca [1 x ptr]
  ; CHECK-NEXT: %[[GUVM:.+]] = alloca ptr
  %guvm = alloca ptr

  ; CHECK-NEXT: %[[CUS:.+]] = call i64 @__kithip_num_cus()
  %cus = call i64 @llvm.kit.gpu.num.compute.units(i32 4)

  ; CHECK-NEXT: %[[STREAM:.+]] = call ptr @__kithip_get_thread_stream()
  %1 = call ptr @llvm.kit.gpu.stream.new(i32 4)

  ; CHECK-NEXT: %[[GSYM:.+]] = call ptr @__kithip_get_global_symbol
  ; CHECK-SAME: (ptr null, ptr @.gname)
  %2 = call ptr @llvm.kit.gpu.symbol.address(i32 4, ptr null, ptr @.gname)

  ; CHECK-NEXT: call void @__kithip_memcpy_htod
  ; CHECK-SAME: (ptr %[[GSYM]], ptr @gbuf, i64 28)
  call void @llvm.kit.gpu.memcpy.htod(i32 4, ptr %2, ptr @gbuf, i64 28)

  ; CHECK-NEXT: call ptr @__kithip_mem_gpu_prefetch
  ; CHECK-SAME: (ptr %[[BUF]], i64 -1, ptr %[[STREAM]])
  %3 = call ptr @llvm.kit.async.gpu.prefetch.htod(i32 4, ptr %buf, i64 -1, ptr %1)

  ; CHECK-NEXT: call ptr @__kithip_mem_gpu_prefetch
  ; CHECK-SAME: (ptr %[[BUF]], i64 1024, ptr %[[STREAM]])
  %4 = call ptr @llvm.kit.async.gpu.prefetch.htod(i32 4, ptr %buf, i64 1024, ptr %1)

  ; CHECK-NEXT: store ptr null, ptr %[[SLOT0]]
  ; CHECK-NEXT: %[[OFF0:.+]] = getelementptr inbounds [1 x ptr], ptr %[[BUNDLE]], i32 0, i32 0
  ; CHECK-NEXT: store ptr %[[SLOT0]], ptr %[[OFF0]]
  ; CHECK-NEXT: call ptr @__kithip_launch_kernel
  ; CHECK-SAME: (ptr null,
  ; CHECK-SAME: ptr @.name,
  ; CHECK-SAME: i64 128,
  ; CHECK-SAME: i64 0,
  ; CHECK-SAME: i64 -1,
  ; CHECK-SAME: i32 24,
  ; CHECK-SAME: ptr null,
  ; CHECK-SAME: ptr %[[STREAM]],
  ; CHECK-SAME: ptr %[[BUNDLE]])
  %5 = call ptr (i32, ptr, ptr, i64, i64, i64, i32, ptr, ptr, ...) @llvm.kit.async.gpu.kernel.launch(i32 4, ptr null, ptr @.name, i64 128, i64 0, i64 -1, i32 24, ptr null, ptr %1, ptr null)

  ; CHECK-NEXT: call void @__kithip_sync_thread_stream(ptr %[[STREAM]])
  call void @llvm.kit.gpu.stream.sync(i32 4, ptr %1)

  ; CHECK-NEXT: call void @__kithip_memcpy_dtoh
  ; CHECK-SAME: (ptr @gbuf, ptr %[[GSYM]], i64 28)
  call void @llvm.kit.gpu.memcpy.dtoh(i32 4, ptr @gbuf, ptr %2, i64 28)

  ; CHECK-NEXT: call ptr @__kithip_mem_host_prefetch
  ; CHECK-SAME: (ptr %[[BUF]], i64 -1, ptr %[[STREAM]])
  %6 = call ptr @llvm.kit.async.gpu.prefetch.dtoh(i32 4, ptr %buf, i64 -1, ptr %1)

  ; CHECK-NEXT: call ptr @__kithip_mem_host_prefetch
  ; CHECK-SAME: (ptr %[[BUF]], i64 1024, ptr %[[STREAM]])
  %7 = call ptr @llvm.kit.async.gpu.prefetch.dtoh(i32 4, ptr %buf, i64 1024, ptr %1)

  ; CHECK-NEXT: %[[MALLOCED:.+]] = call noalias ptr @__kithip_malloc(i64 63)
  %malloced = call noalias ptr @llvm.kit.gpu.malloc(i32 4, i64 63)

  ; CHECK-NEXT: call void @__kithip_memset_bool
  ; CHECK-SAME: (ptr %[[MALLOCED]], i64 63, i8 1)
  call void(i32, ptr, i64, i1, ...) @llvm.kit.gpu.memset.i1(i32 4, ptr %malloced, i64 63, i1 true)

  ; CHECK-NEXT: call void @__kithip_memset_i8
  ; CHECK-SAME: (ptr %[[MALLOCED]], i64 63, i8 1)
  call void(i32, ptr, i64, i8, ...) @llvm.kit.gpu.memset.i8(i32 4, ptr %malloced, i64 63, i8 1)

  ; CHECK-NEXT: call void @__kithip_memset_i16
  ; CHECK-SAME: (ptr %[[MALLOCED]], i64 31, i16 11)
  call void(i32, ptr, i64, i16, ...) @llvm.kit.gpu.memset.i16(i32 4, ptr %malloced, i64 31, i16 11)

  ; CHECK-NEXT: call void @__kithip_memset_i32
  ; CHECK-SAME: (ptr %[[MALLOCED]], i64 15, i32 111)
  call void(i32, ptr, i64, i32, ...) @llvm.kit.gpu.memset.i32(i32 4, ptr %malloced, i64 15, i32 111)

  ; CHECK-NEXT: call void @__kithip_memset_i64
  ; CHECK-SAME: (ptr %[[MALLOCED]], i64 7, i64 1111)
  call void(i32, ptr, i64, i64, ...) @llvm.kit.gpu.memset.i64(i32 4, ptr %malloced, i64 7, i64 1111)

  ; CHECK-NEXT: call void @__kithip_memset_float
  ; CHECK-SAME: (ptr %[[MALLOCED]], i64 15, float 1.000000e+00)
  call void(i32, ptr, i64, float, ...) @llvm.kit.gpu.memset.float(i32 4, ptr %malloced, i64 15, float 1.0)

  ; CHECK-NEXT: call void @__kithip_memset_double
  ; CHECK-SAME: (ptr %[[MALLOCED]], i64 7, double 2.000000e+00)
  call void(i32, ptr, i64, double, ...) @llvm.kit.gpu.memset.double(i32 4, ptr %malloced, i64 7, double 2.0)

  ; CHECK-NEXT: call void @__kithip_memset_from
  ; CHECK-SAME: (ptr %[[MALLOCED]], i64 17, ptr null, i32 48)
  call void(i32, ptr, i64, ptr, ...) @llvm.kit.gpu.memset.ptr(i32 4, ptr %malloced, i64 17, ptr null, i32 48)

  ; CHECK-NEXT: call void @__kithip_free(ptr %[[MALLOCED]])
  call void @llvm.kit.gpu.free(i32 4, ptr %malloced)

  ; CHECK-NEXT: %[[HANDLE:.+]] = call ptr @__kithip_register_devcode(ptr null)
  %handle = call ptr @llvm.kit.gpu.register.devcode(i32 4, ptr null)

  ; CHECK-NEXT: call void @__kithip_register_global
  ; CHECK-SAME: (ptr %[[HANDLE]],
  ; CHECK-SAME: ptr @gbuf,
  ; CHECK-SAME: ptr @.gname,
  ; CHECK-SAME: ptr @.gname,
  ; CHECK-SAME: i64 28,
  ; CHECK-SAME: i32 1,
  ; CHECK-SAME: i32 0
  call void @llvm.kit.gpu.register.global(i32 4, ptr %handle, ptr @gbuf, ptr @.gname, ptr @.gname, i64 28, i32 1, i32 0)

  ; CHECK-NEXT: call void @__kithip_register_global_managed
  ; CHECK-SAME: (ptr %[[HANDLE]],
  ; CHECK-SAME: ptr %[[GUVM]],
  ; CHECK-SAME: ptr @gbuf,
  ; CHECK-SAME: ptr @.gname,
  ; CHECK-SAME: i64 28,
  ; CHECK-SAME: i32 16,
  ; CHECK-SAME: i32 1,
  ; CHECK-SAME: i32 0)
  call void @llvm.kit.gpu.register.global.managed(i32 4, ptr %handle, ptr %guvm, ptr @gbuf, ptr @.gname, i64 28, i32 16, i32 1, i32 0)

  ; CHECK-NEXT: call void @__kithip_unregister_devcode(ptr %[[HANDLE]])
  call void @llvm.kit.gpu.unregister.devcode(i32 4, ptr %handle)

  ; CHECK-NEXT: ret void
  ret void
}
