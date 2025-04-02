; Check that stripmining is done properly on loops over inclusive ranges.
;
; RUN: opt < %s -passes='loop-stripmine' -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, %struct._IO_codecvt*, %struct._IO_wide_data*, %struct._IO_FILE*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type opaque
%struct._IO_codecvt = type opaque
%struct._IO_wide_data = type opaque
%"class.cilk::reducer_opadd" = type { %"class.cilk::reducer.base", [56 x i8] }
%"class.cilk::reducer.base" = type { %"class.cilk::internal::reducer_content.base" }
%"class.cilk::internal::reducer_content.base" = type { %"class.cilk::internal::reducer_base", [56 x i8], [8 x i8] }
%"class.cilk::internal::reducer_base" = type { %struct.__cilkrts_hyperobject_base, %"class.cilk::internal::storage_for_object", i8* }
%struct.__cilkrts_hyperobject_base = type { %struct.cilk_c_monoid, i32, i32, i64 }
%struct.cilk_c_monoid = type { void (i8*, i8*, i8*)*, void (i8*, i8*)*, void (i8*, i8*)*, i8* (%struct.__cilkrts_hyperobject_base*, i64)*, void (%struct.__cilkrts_hyperobject_base*, i8*)* }
%"class.cilk::internal::storage_for_object" = type { %"class.cilk::internal::aligned_storage" }
%"class.cilk::internal::aligned_storage" = type { [1 x i8] }

$__clang_call_terminate = comdat any

$_ZN4cilk8internal12reducer_baseINS_6op_addIxLb1EEEE14reduce_wrapperEPvS5_S5_ = comdat any

$_ZN4cilk8internal12reducer_baseINS_6op_addIxLb1EEEE16identity_wrapperEPvS5_ = comdat any

$_ZN4cilk8internal12reducer_baseINS_6op_addIxLb1EEEE15destroy_wrapperEPvS5_ = comdat any

$_ZN4cilk8internal12reducer_baseINS_6op_addIxLb1EEEE16allocate_wrapperEPvm = comdat any

$_ZN4cilk8internal12reducer_baseINS_6op_addIxLb1EEEE18deallocate_wrapperEPvS5_ = comdat any

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str.12 = private unnamed_addr constant [9 x i8] c"i = %ld\0A\00", align 1
@.str.19 = private unnamed_addr constant [22 x i8] c"this == m_initialThis\00", align 1
@.str.20 = private unnamed_addr constant [74 x i8] c"/data/animals/opencilk/build-test/lib/clang/10.0.1/include/cilk/reducer.h\00", align 1
@__PRETTY_FUNCTION__._ZN4cilk8internal12reducer_baseINS_6op_addIxLb1EEEED2Ev = private unnamed_addr constant [119 x i8] c"cilk::internal::reducer_base<cilk::op_add<long long, true> >::~reducer_base() [Monoid = cilk::op_add<long long, true>]\00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_cilk_reduce_accum.cpp, i8* null }]

declare dso_local void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #0

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nofree nounwind
declare dso_local i32 @__cxa_atexit(void (i8*)*, i8*, i8*) local_unnamed_addr #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #3

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #3

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local i64 @_Z11accum_wrongl(i64 %n) local_unnamed_addr #4 {
entry:
  %accum = alloca i64, align 8
  %syncreg = tail call token @llvm.syncregion.start()
  %accum.0.accum.0..sroa_cast = bitcast i64* %accum to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %accum.0.accum.0..sroa_cast)
  store i64 0, i64* %accum, align 8, !tbaa !2
  %cmp = icmp sgt i64 %n, -1
  br i1 %cmp, label %pfor.cond, label %cleanup

pfor.cond:                                        ; preds = %pfor.inc, %entry
  %__begin.0 = phi i64 [ %inc, %pfor.inc ], [ 0, %entry ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.cond
  %accum.0.load18 = load i64, i64* %accum, align 8
  %add1 = add nsw i64 %accum.0.load18, %__begin.0
  store i64 %add1, i64* %accum, align 8, !tbaa !2
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.cond
  %inc = add nuw i64 %__begin.0, 1
  %exitcond = icmp eq i64 %__begin.0, %n
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.cond, !llvm.loop !6

pfor.cond.cleanup:                                ; preds = %pfor.inc
  sync within %syncreg, label %cleanup

cleanup:                                          ; preds = %pfor.cond.cleanup, %entry
  %accum.0.load19 = load i64, i64* %accum, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %accum.0.accum.0..sroa_cast)
  ret i64 %accum.0.load19
}

; CHECK-LABEL: define {{.*}}i64 @_Z11accum_wrongl(
; CHECK: pfor.cond.preheader:
; CHECK: %[[XTRAITER:.+]] = and i64 %n, 2047
; CHECK-NEXT: add i64 %[[XTRAITER]], 1
; CHECK: %[[CMP:.+]] = icmp slt i64 %n, 2048
; CHECK: br i1 %[[CMP]],

; Function Attrs: argmemonly nounwind willreturn
declare token @llvm.syncregion.start() #3

; Function Attrs: argmemonly willreturn
declare void @llvm.sync.unwind(token) #6

; Function Attrs: nofree nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #7

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8* %0) local_unnamed_addr #8 comdat {
  %2 = tail call i8* @__cxa_begin_catch(i8* %0) #13
  tail call void @_ZSt9terminatev() #16
  unreachable
}

declare dso_local i8* @__cxa_begin_catch(i8*) local_unnamed_addr

declare dso_local void @_ZSt9terminatev() local_unnamed_addr

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) local_unnamed_addr #9

declare dso_local void @__cilkrts_hyper_destroy(%struct.__cilkrts_hyperobject_base*) local_unnamed_addr #0

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdlPv(i8*) local_unnamed_addr #10

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZN4cilk8internal12reducer_baseINS_6op_addIxLb1EEEE14reduce_wrapperEPvS5_S5_(i8* %r, i8* %lhs, i8* %rhs) #5 comdat align 2 {
entry:
  %m_value.i.i = bitcast i8* %rhs to i64*
  %0 = load i64, i64* %m_value.i.i, align 8, !tbaa !25
  %m_value2.i.i = bitcast i8* %lhs to i64*
  %1 = load i64, i64* %m_value2.i.i, align 8, !tbaa !25
  %add.i.i = add nsw i64 %1, %0
  store i64 %add.i.i, i64* %m_value2.i.i, align 8, !tbaa !25
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZN4cilk8internal12reducer_baseINS_6op_addIxLb1EEEE16identity_wrapperEPvS5_(i8* %r, i8* %view) #5 comdat align 2 {
entry:
  %m_value.i.i.i = bitcast i8* %view to i64*
  store i64 0, i64* %m_value.i.i.i, align 8, !tbaa !25
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZN4cilk8internal12reducer_baseINS_6op_addIxLb1EEEE15destroy_wrapperEPvS5_(i8* %r, i8* %view) #5 comdat align 2 {
entry:
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local i8* @_ZN4cilk8internal12reducer_baseINS_6op_addIxLb1EEEE16allocate_wrapperEPvm(i8* %r, i64 %bytes) #5 comdat align 2 {
entry:
  %call.i = tail call i8* @_Znwm(i64 %bytes)
  ret i8* %call.i
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZN4cilk8internal12reducer_baseINS_6op_addIxLb1EEEE18deallocate_wrapperEPvS5_(i8* %r, i8* %view) #5 comdat align 2 {
entry:
  tail call void @_ZdlPv(i8* %view) #13
  ret void
}

declare dso_local void @__cilkrts_hyper_create(%struct.__cilkrts_hyperobject_base*) local_unnamed_addr #0

; Function Attrs: nobuiltin nofree
declare dso_local noalias nonnull i8* @_Znwm(i64) local_unnamed_addr #11

; Function Attrs: nounwind readonly
declare dso_local i8* @__cilkrts_hyper_lookup(%struct.__cilkrts_hyperobject_base*) local_unnamed_addr #12

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_cilk_reduce_accum.cpp() #5 section ".text.startup" {
entry:
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  %0 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* nonnull @__dso_handle) #13
  ret void
}

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nofree nounwind }
attributes #3 = { argmemonly nounwind willreturn }
attributes #4 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { argmemonly willreturn }
attributes #7 = { nofree nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { noinline noreturn nounwind }
attributes #9 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { nobuiltin nofree "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #12 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #13 = { nounwind }
attributes #14 = { cold }
attributes #15 = { nounwind readonly }
attributes #16 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.1 (git@github.com:OpenCilk/opencilk-project.git 46cdea0e962f3423b2fffa5746650aef824098f1)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"long long", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"tapir.loop.spawn.strategy", i32 1}
!8 = !{!9, !10, i64 0}
!9 = !{!"_ZTS13cilk_c_monoid", !10, i64 0, !10, i64 8, !10, i64 16, !10, i64 24, !10, i64 32}
!10 = !{!"any pointer", !4, i64 0}
!11 = !{!9, !10, i64 8}
!12 = !{!9, !10, i64 16}
!13 = !{!9, !10, i64 24}
!14 = !{!9, !10, i64 32}
!15 = !{!16, !17, i64 40}
!16 = !{!"_ZTS26__cilkrts_hyperobject_base", !9, i64 0, !17, i64 40, !17, i64 44, !18, i64 48}
!17 = !{!"int", !4, i64 0}
!18 = !{!"long", !4, i64 0}
!19 = !{!16, !17, i64 44}
!20 = !{!16, !18, i64 48}
!21 = !{!22, !10, i64 64}
!22 = !{!"_ZTSN4cilk8internal12reducer_baseINS_6op_addIxLb1EEEEE", !16, i64 0, !23, i64 56, !10, i64 64}
!23 = !{!"_ZTSN4cilk8internal18storage_for_objectINS_6op_addIxLb1EEEEE"}
!24 = !{!22, !17, i64 44}
!25 = !{!26, !3, i64 0}
!26 = !{!"_ZTSN4cilk11scalar_viewIxEE", !3, i64 0}
!27 = !{!10, !10, i64 0}
!28 = distinct !{!28, !7}
