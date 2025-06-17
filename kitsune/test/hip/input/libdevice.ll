; ModuleID = '/opt/rocm/amdgcn/bitcode/ocml.bc'
source_filename = "llvm-link"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%0 = type { double, double, i32 }
%1 = type { double, double }
%2 = type { double, i32 }
%3 = type { <2 x half>, <2 x i32> }

@__oclc_unsafe_math_opt = external local_unnamed_addr addrspace(4) constant i8, align 1
@__oclc_correctly_rounded_sqrt32 = external local_unnamed_addr addrspace(4) constant i8, align 1

define linkonce_odr hidden noundef float @__ocml_fmuladd_f32(float noundef %0, float noundef %1, float noundef %2) local_unnamed_addr #0 {
  %4 = tail call float @llvm.fmuladd.f32(float %0, float %1, float %2)
  ret float %4
}


; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define linkonce_odr hidden double @__ocml_acos_f64(double noundef %0) local_unnamed_addr #0 {
  %2 = tail call double @llvm.fabs.f64(double %0)
  %3 = fcmp oge double %2, 5.000000e-01
  %4 = tail call double @llvm.fma.f64(double %2, double -5.000000e-01, double 5.000000e-01)
  %5 = fmul double %0, %0
  %6 = select i1 %3, double %4, double %5
  %7 = tail call double @llvm.fma.f64(double %6, double 0x3FA059859FEA6A70, double 0xBF90A5A378A05EAF)
  %8 = tail call double @llvm.fma.f64(double %6, double %7, double 0x3F94052137024D6A)
  %9 = tail call double @llvm.fma.f64(double %6, double %8, double 0x3F7AB3A098A70509)
  %10 = tail call double @llvm.fma.f64(double %6, double %9, double 0x3F88ED60A300C8D2)
  %11 = tail call double @llvm.fma.f64(double %6, double %10, double 0x3F8C6FA84B77012B)
  %12 = tail call double @llvm.fma.f64(double %6, double %11, double 0x3F91C6C111DCCB70)
  %13 = tail call double @llvm.fma.f64(double %6, double %12, double 0x3F96E89F0A0ADACF)
  %14 = tail call double @llvm.fma.f64(double %6, double %13, double 0x3F9F1C72C668963F)
  %15 = tail call double @llvm.fma.f64(double %6, double %14, double 0x3FA6DB6DB41CE4BD)
  %16 = tail call double @llvm.fma.f64(double %6, double %15, double 0x3FB333333336FD5B)
  %17 = tail call double @llvm.fma.f64(double %6, double %16, double 0x3FC5555555555380)
  %18 = fmul double %6, %17
  %19 = tail call double @llvm.fma.f64(double %0, double %18, double %0)
  %20 = fneg double %19
  %21 = tail call double @llvm.fma.f64(double 0x3FEDD9AD336A0500, double 0x3FFAF154EEB562D6, double %20)
  br i1 %3, label %22, label %69

22:                                               ; preds = %1
  %23 = tail call double @llvm.amdgcn.rsq.f64(double %4)
  %24 = fmul double %4, %23
  %25 = fmul double %23, 5.000000e-01
  %26 = fneg double %25
  %27 = tail call double @llvm.fma.f64(double %26, double %24, double 5.000000e-01)
  %28 = tail call double @llvm.fma.f64(double %25, double %27, double %25)
  %29 = tail call double @llvm.fma.f64(double %24, double %27, double %24)
  %30 = fneg double %29
  %31 = tail call double @llvm.fma.f64(double %30, double %29, double %4)
  %32 = tail call double @llvm.fma.f64(double %31, double %28, double %29)
  %33 = fcmp oeq double %4, 0.000000e+00
  %34 = select i1 %33, double %4, double %32
  %35 = fmul double %34, %34
  %36 = fneg double %35
  %37 = tail call double @llvm.fma.f64(double %34, double %34, double %36)
  %38 = fsub double %4, %35
  %39 = fsub double %4, %38
  %40 = fsub double %39, %35
  %41 = fsub double %40, %37
  %42 = fadd double %38, %41
  %43 = fmul double %34, 2.000000e+00
  %44 = tail call double @llvm.amdgcn.rcp.f64(double %43)
  %45 = fneg double %43
  %46 = tail call double @llvm.fma.f64(double %45, double %44, double 1.000000e+00)
  %47 = tail call double @llvm.fma.f64(double %46, double %44, double %44)
  %48 = tail call double @llvm.fma.f64(double %45, double %47, double 1.000000e+00)
  %49 = tail call double @llvm.fma.f64(double %48, double %47, double %47)
  %50 = fmul double %42, %49
  %51 = tail call double @llvm.fma.f64(double %45, double %50, double %42)
  %52 = tail call double @llvm.fma.f64(double %51, double %49, double %50)
  %53 = select i1 %33, double 0.000000e+00, double %52
  %54 = fadd double %34, %53
  %55 = fsub double %54, %34
  %56 = fsub double %53, %55
  %57 = tail call double @llvm.fma.f64(double %54, double %18, double %54)
  %58 = fmul double %57, -2.000000e+00
  %59 = tail call double @llvm.fma.f64(double 0x3FFDD9AD336A0500, double 0x3FFAF154EEB562D6, double %58)
  %60 = tail call double @llvm.fma.f64(double %54, double %18, double %56)
  %61 = fadd double %54, %60
  %62 = fmul double %61, 2.000000e+00
  %63 = fcmp olt double %0, 0.000000e+00
  %64 = select i1 %63, double %59, double %62
  %65 = fcmp oeq double %0, -1.000000e+00
  %66 = select i1 %65, double 0x400921FB54442D18, double %64
  %67 = fcmp oeq double %0, 1.000000e+00
  %68 = select i1 %67, double 0.000000e+00, double %66
  br label %69

69:                                               ; preds = %22, %1
  %70 = phi double [ %68, %22 ], [ %21, %1 ]
  ret double %70
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(none)
define linkonce_odr hidden float @__ocml_acos_f32(float noundef %0) local_unnamed_addr #2 {
  %2 = tail call float @llvm.fabs.f32(float %0)
  %3 = tail call float @__ocml_fmuladd_f32(float noundef -5.000000e-01, float noundef %2, float noundef 5.000000e-01) #15
  %4 = fmul float %0, %0
  %5 = fcmp ogt float %2, 5.000000e-01
  %6 = select i1 %5, float %3, float %4
  %7 = tail call float @__ocml_fmuladd_f32(float noundef %6, float noundef 0x3FA38434E0000000, float noundef 0x3F8BF8BB40000000) #15
  %8 = tail call float @__ocml_fmuladd_f32(float noundef %6, float noundef %7, float noundef 0x3FA0698780000000) #15
  %9 = tail call float @__ocml_fmuladd_f32(float noundef %6, float noundef %8, float noundef 0x3FA6C83620000000) #15
  %10 = tail call float @__ocml_fmuladd_f32(float noundef %6, float noundef %9, float noundef 0x3FB3337900000000) #15
  %11 = tail call float @__ocml_fmuladd_f32(float noundef %6, float noundef %10, float noundef 0x3FC5555580000000) #15
  %12 = fmul float %6, %11
  %13 = tail call float @llvm.amdgcn.sqrt.f32(float %6)
  %14 = tail call float @__ocml_fmuladd_f32(float noundef %13, float noundef %12, float noundef %13) #15
  %15 = fmul float %14, 2.000000e+00
  %16 = fneg float %15
  %17 = tail call float @__ocml_fmuladd_f32(float noundef 0x3FFDDCB020000000, float noundef 0x3FFAEE9D60000000, float noundef %16) #15
  %18 = fcmp olt float %0, 0.000000e+00
  %19 = select i1 %18, float %17, float %15
  %20 = tail call float @__ocml_fmuladd_f32(float noundef %0, float noundef %12, float noundef %0) #15
  %21 = fneg float %20
  %22 = tail call float @__ocml_fmuladd_f32(float noundef 0x3FEDDCB020000000, float noundef 0x3FFAEE9D60000000, float noundef %21) #15
  %23 = select i1 %5, float %19, float %22
  ret float %23
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define linkonce_odr hidden noundef double @__ocml_sqrt_f64(double noundef %0) local_unnamed_addr #0 {
  %2 = tail call double @llvm.sqrt.f64(double %0)
  ret double %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define linkonce_odr hidden float @__ocml_sqrt_f32(float noundef %0) local_unnamed_addr #0 {
  %2 = load i8, ptr addrspace(4) @__oclc_correctly_rounded_sqrt32, align 1, !tbaa !4, !range !8, !noundef !9
  %3 = trunc nuw i8 %2 to i1
  br i1 %3, label %4, label %42

4:                                                ; preds = %1
  %5 = fcmp olt float %0, 0x39F0000000000000
  %6 = select i1 %5, float 0x41F0000000000000, float 1.000000e+00
  %7 = fmul float %6, %0
  %8 = tail call float @llvm.canonicalize.f32(float 0x36A0000000000000)
  %9 = tail call i1 @llvm.is.fpclass.f32(float %8, i32 64)
  br i1 %9, label %25, label %10

10:                                               ; preds = %4
  %11 = tail call float @llvm.amdgcn.sqrt.f32(float %7)
  %12 = bitcast float %11 to i32
  %13 = add nsw i32 %12, -1
  %14 = bitcast i32 %13 to float
  %15 = add nsw i32 %12, 1
  %16 = bitcast i32 %15 to float
  %17 = fneg float %14
  %18 = tail call float @llvm.fma.f32(float %17, float %11, float %7)
  %19 = fneg float %16
  %20 = tail call float @llvm.fma.f32(float %19, float %11, float %7)
  %21 = fcmp ole float %18, 0.000000e+00
  %22 = select i1 %21, float %14, float %11
  %23 = fcmp ogt float %20, 0.000000e+00
  %24 = select i1 %23, float %16, float %22
  br label %36

25:                                               ; preds = %4
  %26 = tail call float @llvm.amdgcn.rsq.f32(float %7)
  %27 = fmul float %7, %26
  %28 = fmul float %26, 5.000000e-01
  %29 = fneg float %28
  %30 = tail call float @llvm.fma.f32(float %29, float %27, float 5.000000e-01)
  %31 = tail call float @llvm.fma.f32(float %28, float %30, float %28)
  %32 = tail call float @llvm.fma.f32(float %27, float %30, float %27)
  %33 = fneg float %32
  %34 = tail call float @llvm.fma.f32(float %33, float %32, float %7)
  %35 = tail call float @llvm.fma.f32(float %34, float %31, float %32)
  br label %36

36:                                               ; preds = %25, %10
  %37 = phi float [ %35, %25 ], [ %24, %10 ]
  %38 = select i1 %5, float 0x3EF0000000000000, float 1.000000e+00
  %39 = fmul float %38, %37
  %40 = tail call i1 @llvm.is.fpclass.f32(float %7, i32 608)
  %41 = select i1 %40, float %7, float %39
  br label %54

42:                                               ; preds = %1
  %43 = tail call float @llvm.canonicalize.f32(float 0x36A0000000000000)
  %44 = tail call i1 @llvm.is.fpclass.f32(float %43, i32 64)
  br i1 %44, label %45, label %47

45:                                               ; preds = %42
  %46 = tail call float @llvm.amdgcn.sqrt.f32(float %0)
  br label %54

47:                                               ; preds = %42
  %48 = fcmp olt float %0, 0x3810000000000000
  %49 = tail call float @llvm.ldexp.f32.i32(float %0, i32 32)
  %50 = select i1 %48, float %49, float %0
  %51 = tail call float @llvm.amdgcn.sqrt.f32(float %50)
  %52 = tail call float @llvm.ldexp.f32.i32(float %51, i32 -16)
  %53 = select i1 %48, float %52, float %51
  br label %54

54:                                               ; preds = %47, %45, %36
  %55 = phi float [ %41, %36 ], [ %46, %45 ], [ %53, %47 ]
  ret float %55
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "denormal-fp-math"="dynamic,dynamic" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nofree norecurse nounwind willreturn memory(none) "denormal-fp-math"="dynamic,dynamic" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind strictfp willreturn memory(none) "denormal-fp-math"="dynamic,dynamic" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nocallback nofree nosync nounwind willreturn }
attributes #5 = { nocallback nofree nosync nounwind strictfp willreturn memory(inaccessiblemem: readwrite) }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: write) "denormal-fp-math"="dynamic,dynamic" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #8 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) "denormal-fp-math"="dynamic,dynamic" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #9 = { convergent norecurse nounwind "denormal-fp-math"="dynamic,dynamic" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #10 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "approx-func-fp-math"="true" "denormal-fp-math"="dynamic,dynamic" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #11 = { nofree norecurse nosync nounwind memory(none) "denormal-fp-math"="dynamic,dynamic" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #12 = { nofree norecurse nosync nounwind memory(argmem: write) "denormal-fp-math"="dynamic,dynamic" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #13 = { convergent nofree norecurse nounwind memory(none) "denormal-fp-math"="dynamic,dynamic" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #14 = { convergent nofree norecurse nounwind memory(argmem: write) "denormal-fp-math"="dynamic,dynamic" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #15 = { convergent nounwind willreturn memory(none) }
attributes #16 = { strictfp }
attributes #17 = { nounwind }
attributes #18 = { convergent nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 19.0.0git (/srcdest/rocm-llvm c7fe45cf4b819c5991fe208aaa96edf142730f1d)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"bool", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{i8 0, i8 2}
!9 = !{}
!10 = !{float 2.500000e+00}
!11 = !{!12, !12, i64 0}
!12 = !{!"double", !6, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"float", !6, i64 0}
!15 = !{!6, !6, i64 0}
!16 = !{!17, !17, i64 0}
!17 = !{!"half", !6, i64 0}
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !6, i64 0}
!20 = !{float 3.000000e+00}
