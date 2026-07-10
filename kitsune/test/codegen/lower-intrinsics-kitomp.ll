; Check that intrinsics that map to Kitsune's openmp runtime are lowered
; correctly. If more intrinsics are created, they should be added here to test
; basic intrinsic lowering.
;
; We also check that any attributes attached to the arguments and the call
; itself are also attached to the lowered call.
;
; RUN: opt --tapir=openmp -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: call void @__kitomp_initialize() #[[INITIALIZE:[0-9]+]]
; CHECK-NEXT: call void @__kitrt_enable_verbose_mode() #[[VERBOSE:[0-9]+]]
; CHECK-NEXT: call void @__kitomp_launch(ptr nonnull @f, i64 0, i64 128, ptr @gbuf) #[[LAUNCH:[0-9]+]]
; CHECK-NEXT: call i32 @__kitomp_num_threads() #[[THREADS:[0-9]+]]
; CHECK-NEXT: call i64 @__kitomp_reduce_num_partials(i64 %[[N]]) #[[PARTIALS:[0-9]+]]
; CHECK-NEXT: call void @__kitomp_finalize() #[[FINALIZE:[0-9]+]]
;
; CHECK-DAG: #[[INITIALIZE]] = { "initialize" }
; CHECK-DAG: #[[VERBOSE]] = { "verbose" }
; CHECK-DAG: #[[LAUNCH]] = { "launch" }
; CHECK-DAG: #[[THREADS]] = { "threads" }
; CHECK-DAG: #[[PARTIALS]] = { "partials" }
; CHECK-DAG: #[[FINALIZE]] = { "finalize" }

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]

define void @f(ptr %buf, i64 %n) {
  call void @llvm.kit.runtime.initialize(i32 512) #0
  call void @llvm.kit.runtime.set.verbose(i32 2, i8 1) #1
  call void @llvm.kit.runtime.set.verbose(i32 2, i8 0) #1
  call void @llvm.kit.cpu.threads.launch(i32 512, ptr nonnull @f, i64 0, i64 128, ptr @gbuf) #2
  %numThreads = call i32 @llvm.kit.cpu.num.threads(i32 512) #3
  %numPartials = call i64 @llvm.kit.reduce.num.partials(i32 512, i64 %n) #4
  call void @llvm.kit.runtime.finalize(i32 512) #5
  ret void
}

attributes #0 = { "initialize" }
attributes #1 = { "verbose" }
attributes #2 = { "launch" }
attributes #3 = { "threads" }
attributes #4 = { "partials" }
attributes #5 = { "finalize" }
