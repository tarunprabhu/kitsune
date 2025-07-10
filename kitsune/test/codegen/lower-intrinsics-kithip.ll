; REQUIRES: kitsune-hip
;
; Check that intrinsics for Kitsune's hip runtime are lowered correctly.
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=hip -passes='kit-lower-intrinsics' -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: %1 = alloca ptr
; CHECK-NEXT: %2 = alloca [1 x ptr]
; CHECK-NEXT: call void @__kithip_initialize()
; CHECK-NEXT: call void @__kithip_enable_ylaunch()
; CHECK-NOT: call void @__kithip_enable_ylaunch()
; CHECK-NEXT: call void @__kithip_enable_xnack()
; CHECK-NOT: call void @__kithip_enable_xnack()
; CHECK-NEXT: call void @__kithip_set_default_threads_per_blk(i32 24)
; CHECK-NEXT: call void @__kithip_set_max_threads_per_blk(i32 1024)
; CHECK-NEXT: %3 = call ptr @__kithip_get_thread_stream()
; CHECK-NEXT: %4 = call ptr @__kithip_get_global_symbol(ptr null, ptr @.gname)
; CHECK-NEXT: call void @__kithip_memcpy_sym_to_device(ptr @gbuf, ptr %4, i64 28)
; CHECK-NEXT: %5 = call ptr @__kithip_mem_gpu_prefetch(ptr %buf, ptr %3)
; CHECK-NEXT: %6 = call ptr @__kithip_mem_gpu_prefetch(ptr %buf, ptr %3)
; CHECK-NEXT: store ptr null, ptr %1
; CHECK-NEXT: %7 = getelementptr inbounds [1 x ptr], ptr %2, i64 0, i64 0
; CHECK-NEXT: store ptr %1, ptr %7
; CHECK-NEXT: %8 = call ptr @__kithip_launch_kernel(ptr null, ptr @.name, ptr nonnull %2, i64 128, i32 24, ptr null, ptr %3)
; CHECK-NEXT: call void @__kithip_sync_thread_stream(ptr %3)
; CHECK-NEXT: call void @__kithip_memcpy_sym_to_host(ptr %4, ptr @gbuf, i64 28)
; CHECK-NEXT: %9 = call ptr @__kithip_mem_host_prefetch(ptr %buf, ptr %3)
; CHECK-NEXT: %10 = call ptr @__kithip_mem_host_prefetch(ptr %buf, ptr %3)
; CHECK-NEXT: call void @__kithip_destroy()
; CHECK-NEXT: ret void
;
; CHECK-DAG: void @__kithip_destroy() #[[ATTRS:[0-9]+]]
; CHECK-DAG: void @__kithip_enable_xnack() #[[ATTRS]]
; CHECK-DAG: void @__kithip_enable_ylaunch() #[[ATTRS]]
; CHECK-DAG: ptr @__kithip_get_global_symbol(ptr, ptr) #[[ATTRS]]
; CHECK-DAG: void @__kithip_initialize() #[[ATTRS]]
; CHECK-DAG: ptr @__kithip_launch_kernel(ptr, ptr, ptr, i64, i32, ptr, ptr) #[[ATTRS]]
; CHECK-DAG: ptr @__kithip_mem_gpu_prefetch(ptr, ptr) #[[ATTRS]]
; CHECK-DAG: ptr @__kithip_mem_host_prefetch(ptr, ptr) #[[ATTRS]]
; CHECK-DAG: void @__kithip_memcpy_sym_to_device(ptr, ptr, i64) #[[ATTRS]]
; CHECK-DAG: void @__kithip_memcpy_sym_to_host(ptr, ptr, i64) #[[ATTRS]]
; CHECK-DAG: void @__kithip_set_default_threads_per_blk(i32) #[[ATTRS]]
; CHECK-DAG: void @__kithip_set_max_threads_per_blk(i32) #[[ATTRS]]
; CHECK-DAG: void @__kithip_sync_thread_stream(ptr) #[[ATTRS]]
;
; CHECK-DAG: #[[ATTRS]] = { nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) }

target triple = "x86_64-unknown-linux-gnu"

@gbuf = external global [7 x float]
@.gname = unnamed_addr constant [5 x i8] c "gbuf\00"
@.name = unnamed_addr constant [7 x i8] c"kernel\00"

define void @f(ptr %buf, i64 %n) {
  call void @llvm.kit.initialize(i32 4)
  call void @llvm.kit.enable.y.axis.launches(i32 4, i8 1)
  call void @llvm.kit.enable.y.axis.launches(i32 4, i8 0)
  call void @llvm.kit.enable.xnack(i8 42)
  call void @llvm.kit.enable.xnack(i8 0)
  call void @llvm.kit.set.fixed.tpb(i32 4, i32 24)
  call void @llvm.kit.set.max.tpb(i32 4, i32 1024)
  %1 = call ptr @llvm.kit.thread.stream(i32 4)
  %2 = call ptr @llvm.kit.symbol.device.ptr(i32 4, ptr null, ptr @.gname)
  call void @llvm.kit.symbol.memcpy.htod(i32 4, ptr %2, ptr @gbuf, i64 28)
  %3 = call ptr @llvm.kit.async.prefetch.htod(i32 4, ptr %buf, i64 -1, ptr %1)
  %4 = call ptr @llvm.kit.async.prefetch.htod(i32 4, ptr %buf, i64 1024, ptr %1)
  %5 = call ptr (i32, ptr, ptr, i64, i32, ptr, ptr, ...) @llvm.kit.async.launch.kernel(i32 4, ptr null, ptr @.name, i64 128, i32 24, ptr null, ptr %1, ptr null)
  call void @llvm.kit.sync.stream(i32 4, ptr %1)
  call void @llvm.kit.symbol.memcpy.dtoh(i32 4, ptr @gbuf, ptr %2, i64 28)
  %6 = call ptr @llvm.kit.async.prefetch.dtoh(i32 4, ptr %buf, i64 -1, ptr %1)
  %7 = call ptr @llvm.kit.async.prefetch.dtoh(i32 4, ptr %buf, i64 1024, ptr %1)
  call void @llvm.kit.finalize(i32 4)
  ret void
}
