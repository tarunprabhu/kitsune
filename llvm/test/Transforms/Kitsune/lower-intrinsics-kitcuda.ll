; RUN: opt -passes='lower-kitsune-runtime-intrinsics' -S %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@gbuf = external global [7 x float]
@.gname = unnamed_addr constant [5 x i8] c "gbuf\00"
@.name = unnamed_addr constant [7 x i8] c"kernel\00"

declare void @llvm.kitrt.enable.refine.launches(i8, i8)
declare void @llvm.kitrt.finalize(i8)
declare void @llvm.kitrt.initialize(i8)
declare ptr @llvm.kitrt.launch.kernel(i8, ptr, ptr, ptr, i64, i32, ptr, ptr)
declare ptr @llvm.kitrt.prefetch.device(i8, ptr, i64, ptr)
declare ptr @llvm.kitrt.prefetch.host(i8, ptr, i64, ptr)
declare void @llvm.kitrt.set.fixed.tpb(i8, i32)
declare void @llvm.kitrt.set.max.tpb(i8, i32)
declare ptr @llvm.kitrt.symbol.device.ptr(i8, ptr, ptr)
declare void @llvm.kitrt.symbol.memcpy.device(i8, ptr, ptr, i64)
declare void @llvm.kitrt.symbol.memcpy.host(i8, ptr, ptr, i64)
declare void @llvm.kitrt.sync.stream(i8, ptr)

; Function Attrs: nounwind memory(inaccessiblemem: readwrite) uwtable
define dso_local void @f(ptr noundef %buf, i64 noundef %n) local_unnamed_addr #0 {
entry:
  call void @llvm.kitrt.initialize(i8 2)
  call void @llvm.kitrt.enable.refine.launches(i8 2, i8 1)
  call void @llvm.kitrt.enable.refine.launches(i8 2, i8 0)
  call void @llvm.kitrt.set.fixed.tpb(i8 2, i32 24)
  call void @llvm.kitrt.set.max.tpb(i8 2, i32 1024)
  %0 = call ptr @llvm.kitrt.symbol.device.ptr(i8 2, ptr null, ptr @.gname)
  call void @llvm.kitrt.symbol.memcpy.device(i8 2, ptr %0, ptr @gbuf, i64 28)
  %1 = call ptr @llvm.kitrt.prefetch.device(i8 2, ptr %buf, i64 -1, ptr null)
  %2 = call ptr @llvm.kitrt.prefetch.device(i8 2, ptr %buf, i64 1024, ptr %1)
  %3 = call ptr @llvm.kitrt.launch.kernel(i8 2, ptr null, ptr @.name, ptr null, i64 128, i32 24, ptr null, ptr %2)
  call void @llvm.kitrt.sync.stream(i8 2, ptr %3)
  call void @llvm.kitrt.symbol.memcpy.host(i8 2, ptr @gbuf, ptr %0, i64 28)
  %4 = call ptr @llvm.kitrt.prefetch.host(i8 2, ptr %buf, i64 -1, ptr %3)
  %5 = call ptr @llvm.kitrt.prefetch.host(i8 2, ptr %buf, i64 1024, ptr %4)
  call void @llvm.kitrt.finalize(i8 2)
  ret void
}

attributes #0 = { nounwind uwtable }

; CHECK-LABEL: @f
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @__kitcuda_initialize()
; CHECK-NEXT: call void @__kitcuda_enable_launch_refinement(i8 1)
; CHECK-NEXT: call void @__kitcuda_enable_launch_refinement(i8 0)
; CHECK-NEXT: call void @__kitcuda_set_default_threads_per_blk(i32 24)
; CHECK-NEXT: call void @__kitcuda_set_max_threads_per_blk(i32 1024)
; CHECK-NEXT: %0 = call ptr @__kitcuda_get_global_symbol(ptr null, ptr @.gname)
; CHECK-NEXT: call void @__kitcuda_memcpy_sym_to_device(ptr @gbuf, ptr %0, i64 28)
; CHECK-NEXT: %1 = call ptr @__kitcuda_mem_gpu_prefetch(ptr %buf, ptr null)
; CHECK-NEXT: %2 = call ptr @__kitcuda_mem_gpu_prefetch(ptr %buf, ptr %1)
; CHECK-NEXT: %3 = call ptr @__kitcuda_launch_kernel(ptr null, ptr @.name, ptr null, i64 128, i32 24, ptr null, ptr %2)
; CHECK-NEXT: call void @__kitcuda_sync_thread_stream(ptr %3)
; CHECK-NEXT: call void @__kitcuda_memcpy_sym_to_host(ptr %0, ptr @gbuf, i64 28)
; CHECK-NEXT: %4 = call ptr @__kitcuda_mem_host_prefetch(ptr %buf, ptr %3)
; CHECK-NEXT: %5 = call ptr @__kitcuda_mem_host_prefetch(ptr %buf, ptr %4)
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