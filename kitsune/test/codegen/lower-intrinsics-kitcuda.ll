; REQUIRES: kitsune-cuda
;
; Check that intrinsics that map to Kitsune's cuda runtime are lowered
; correctly. If more intrinsics are created, they should be added here so that
; the basic lowering can be tested.
;
; RUN: opt --tapir=cuda -passes='kit-lower-intrinsics' -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: %1 = alloca ptr
; CHECK-NEXT: %2 = alloca [1 x ptr]
; CHECK-NEXT: call void @__kitcuda_initialize()
; CHECK-NEXT: call void @__kitcuda_enable_launch_refinement(i8 1)
; CHECK-NEXT: call void @__kitcuda_enable_launch_refinement(i8 0)
; CHECK-NEXT: call void @__kitcuda_set_default_threads_per_blk(i32 24)
; CHECK-NEXT: call void @__kitcuda_set_max_threads_per_blk(i32 1024)
; CHECK-NEXT: %3 = call ptr @__kitcuda_get_global_symbol(ptr null, ptr @.gname)
; CHECK-NEXT: call void @__kitcuda_memcpy_sym_to_device(ptr @gbuf, ptr %3, i64 28)
; CHECK-NEXT: %4 = call ptr @__kitcuda_mem_gpu_prefetch(ptr %buf, ptr null)
; CHECK-NEXT: %5 = call ptr @__kitcuda_mem_gpu_prefetch(ptr %buf, ptr %4)
; CHECK-NEXT: store ptr null, ptr %1
; CHECK-NEXT: %6 = getelementptr inbounds [1 x ptr], ptr %2, i64 0, i64 0
; CHECK-NEXT: store ptr %1, ptr %6
; CHECK-NEXT: %7 = call ptr @__kitcuda_launch_kernel(ptr null, ptr @.name, ptr nonnull %2, i64 128, i32 24, ptr null, ptr %5)
; CHECK-NEXT: call void @__kitcuda_sync_thread_stream(ptr %7)
; CHECK-NEXT: call void @__kitcuda_memcpy_sym_to_host(ptr %3, ptr @gbuf, i64 28)
; CHECK-NEXT: %8 = call ptr @__kitcuda_mem_host_prefetch(ptr %buf, ptr %7)
; CHECK-NEXT: %9 = call ptr @__kitcuda_mem_host_prefetch(ptr %buf, ptr %8)
; CHECK-NEXT: call void @__kitcuda_destroy()
; CHECK-NEXT: ret void
;
; CHECK-DAG: void @__kitcuda_destroy() #[[ATTRS:[0-9]+]]
; CHECK-DAG: void @__kitcuda_enable_launch_refinement(i8) #[[ATTRS]]
; CHECK-DAG: ptr @__kitcuda_get_global_symbol(ptr, ptr) #[[ATTRS]]
; CHECK-DAG: void @__kitcuda_initialize() #[[ATTRS]]
; CHECK-DAG: ptr @__kitcuda_launch_kernel(ptr, ptr, ptr, i64, i32, ptr, ptr) #[[ATTRS]]
; CHECK-DAG: ptr @__kitcuda_mem_gpu_prefetch(ptr, ptr) #[[ATTRS]]
; CHECK-DAG: ptr @__kitcuda_mem_host_prefetch(ptr, ptr) #[[ATTRS]]
; CHECK-DAG: void @__kitcuda_memcpy_sym_to_device(ptr, ptr, i64) #[[ATTRS]]
; CHECK-DAG: void @__kitcuda_memcpy_sym_to_host(ptr, ptr, i64) #[[ATTRS]]
; CHECK-DAG: void @__kitcuda_set_default_threads_per_blk(i32) #[[ATTRS]]
; CHECK-DAG: void @__kitcuda_set_max_threads_per_blk(i32) #[[ATTRS]]
; CHECK-DAG: void @__kitcuda_sync_thread_stream(ptr) #[[ATTRS]]
;
; CHECK-DAG: #[[ATTRS]] = { nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) }

target triple = "x86_64-unknown-linux-gnu"

@gbuf = external global [7 x float]
@.gname = unnamed_addr constant [5 x i8] c "gbuf\00"
@.name = unnamed_addr constant [7 x i8] c"kernel\00"

define void @f(ptr %buf, i64 %n) {
  call void @llvm.kit.initialize(i32 2)
  call void @llvm.kit.enable.refine.launches(i32 2, i8 1)
  call void @llvm.kit.enable.refine.launches(i32 2, i8 0)
  call void @llvm.kit.set.fixed.tpb(i32 2, i32 24)
  call void @llvm.kit.set.max.tpb(i32 2, i32 1024)
  %1 = call ptr @llvm.kit.symbol.device.ptr(i32 2, ptr null, ptr @.gname)
  call void @llvm.kit.symbol.memcpy.htod(i32 2, ptr %1, ptr @gbuf, i64 28)
  %2 = call ptr @llvm.kit.async.prefetch.htod(i32 2, ptr %buf, i64 -1, ptr null)
  %3 = call ptr @llvm.kit.async.prefetch.htod(i32 2, ptr %buf, i64 1024, ptr %2)
  %4 = call ptr (i32, ptr, ptr, i64, i32, ptr, ptr, ...) @llvm.kit.async.launch.kernel(i32 2, ptr null, ptr @.name, i64 128, i32 24, ptr null, ptr %3, ptr null)
  call void @llvm.kit.sync.stream(i32 2, ptr %4)
  call void @llvm.kit.symbol.memcpy.dtoh(i32 2, ptr @gbuf, ptr %1, i64 28)
  %5 = call ptr @llvm.kit.async.prefetch.dtoh(i32 2, ptr %buf, i64 -1, ptr %4)
  %6 = call ptr @llvm.kit.async.prefetch.dtoh(i32 2, ptr %buf, i64 1024, ptr %5)
  call void @llvm.kit.finalize(i32 2)
  ret void
}
