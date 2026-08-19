; REQUIRES: kitsune-cuda
;
; Check that intrinsics that map to Kitsune's cuda runtime are lowered
; correctly. If more intrinsics are created, they should be added here to test
; basic intrinsic lowering.
;
; RUN: opt --tapir=cuda -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: %[[SLOT0:.+]] = alloca ptr
; CHECK-NEXT: %[[BUNDLE:.+]] = alloca [1 x ptr]
; CHECK-NEXT: %guvm = alloca ptr
; CHECK-NEXT: %[[CUS:.+]] = call i64 @__kitcuda_num_sms()
; CHECK-NEXT: %[[STREAM:.+]] = call ptr @__kitcuda_get_thread_stream()
; CHECK-NEXT: %[[GSYM:.+]] = call ptr @__kitcuda_get_global_symbol(ptr null, ptr @.gname)
; CHECK-NEXT: %[[GSYMI:.+]] = ptrtoint ptr %[[GSYM]] to i64
; CHECK-NEXT: call void @__kitcuda_memcpy_htod(i64 %[[GSYMI]], ptr @gbuf, i64 28)
; CHECK-NEXT: call ptr @__kitcuda_mem_gpu_prefetch(ptr %[[BUF]], i64 -1, ptr %[[STREAM]])
; CHECK-NEXT: call ptr @__kitcuda_mem_gpu_prefetch(ptr %[[BUF]], i64 1024, ptr %[[STREAM]])
; CHECK-NEXT: store ptr null, ptr %[[SLOT0]]
; CHECK-NEXT: %[[OFF0:.+]] = getelementptr inbounds [1 x ptr], ptr %[[BUNDLE]], i32 0, i32 0
; CHECK-NEXT: store ptr %[[SLOT0]], ptr %[[OFF0]]
; CHECK-NEXT: call ptr @__kitcuda_launch_kernel(ptr null, ptr @.name, i64 128, i64 0, i64 -1, i32 24, ptr null, ptr %[[STREAM]], ptr %[[BUNDLE]])
; CHECK-NEXT: call void @__kitcuda_sync_thread_stream(ptr %[[STREAM]])
; CHECK-NEXT: %[[GSYMI:.+]] = ptrtoint ptr %[[GSYM]] to i64
; CHECK-NEXT: call void @__kitcuda_memcpy_dtoh(ptr @gbuf, i64 %[[GSYMI]], i64 28)
; CHECK-NEXT: call ptr @__kitcuda_mem_host_prefetch(ptr %[[BUF]], i64 -1, ptr %[[STREAM]])
; CHECK-NEXT: call ptr @__kitcuda_mem_host_prefetch(ptr %[[BUF]], i64 1024, ptr %[[STREAM]])
; CHECK-NEXT: %[[MALLOCED:.+]] = call noalias ptr @__kitcuda_malloc(i64 63)
; CHECK-NEXT: call void @__kitcuda_free(ptr %[[MALLOCED]])
; CHECK-NEXT: %[[HANDLE:.+]] = call ptr @__kitcuda_register_devcode(ptr null)
; CHECK-NEXT: call void @__kitcuda_register_devcode_end(ptr %[[HANDLE]])
; CHECK-NEXT: call void @__kitcuda_register_global(ptr %[[HANDLE]], ptr @gbuf, ptr @.gname, ptr @.gname, i64 28, i32 1, i32 0)
; CHECK-NEXT: call void @__kitcuda_register_global_managed(ptr %[[HANDLE]], ptr %guvm, ptr @gbuf, ptr @.gname, i64 28, i32 16, i32 1, i32 0)
; CHECK-NEXT: call void @__kitcuda_unregister_devcode(ptr %[[HANDLE]])
; CHECK-NEXT: ret void

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]
@.gname = unnamed_addr constant [5 x i8] c "gbuf\00"
@.name = unnamed_addr constant [7 x i8] c"kernel\00"

define void @f(ptr %buf, i64 %n) {
  %guvm = alloca ptr
  %cus = call i64 @llvm.kit.gpu.num.compute.units(i32 2)
  %1 = call ptr @llvm.kit.gpu.stream.new(i32 2)
  %2 = call ptr @llvm.kit.gpu.symbol.address(i32 2, ptr null, ptr @.gname)
  call void @llvm.kit.gpu.memcpy.htod(i32 2, ptr %2, ptr @gbuf, i64 28)
  %3 = call ptr @llvm.kit.async.gpu.prefetch.htod(i32 2, ptr %buf, i64 -1, ptr %1)
  %4 = call ptr @llvm.kit.async.gpu.prefetch.htod(i32 2, ptr %buf, i64 1024, ptr %1)
  %5 = call ptr (i32, ptr, ptr, i64, i64, i64, i32, ptr, ptr, ...) @llvm.kit.async.gpu.kernel.launch(i32 2, ptr null, ptr @.name, i64 128, i64 0, i64 -1, i32 24, ptr null, ptr %1, ptr null)
  call void @llvm.kit.gpu.stream.sync(i32 2, ptr %1)
  call void @llvm.kit.gpu.memcpy.dtoh(i32 2, ptr @gbuf, ptr %2, i64 28)
  %6 = call ptr @llvm.kit.async.gpu.prefetch.dtoh(i32 2, ptr %buf, i64 -1, ptr %1)
  %7 = call ptr @llvm.kit.async.gpu.prefetch.dtoh(i32 2, ptr %buf, i64 1024, ptr %1)
  %malloced = call noalias ptr @llvm.kit.gpu.malloc(i32 2, i64 63)
  call void @llvm.kit.gpu.free(i32 2, ptr %malloced)
  %handle = call ptr @llvm.kit.gpu.register.devcode(i32 2, ptr null)
  call void @llvm.kit.gpu.register.devcode.end(i32 2, ptr %handle)
  call void @llvm.kit.gpu.register.global(i32 2, ptr %handle, ptr @gbuf, ptr @.gname, ptr @.gname, i64 28, i32 1, i32 0)
  call void @llvm.kit.gpu.register.global.managed(i32 2, ptr %handle, ptr %guvm, ptr @gbuf, ptr @.gname, i64 28, i32 16, i32 1, i32 0)
  call void @llvm.kit.gpu.unregister.devcode(i32 2, ptr %handle)
  ret void
}
