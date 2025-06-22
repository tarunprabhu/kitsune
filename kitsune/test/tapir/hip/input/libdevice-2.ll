; ---------------------- THIS IS INTENTIONALLY INCOMPLETE ----------------------
;
; @__ocml_fmuladd_f32 is not defined here and any attempt to use LLVM tools on
; this will fail the verifier. This is intention because this is intended to
; test that when linking hte libdevice bitcode into the kernel module, the LLVM
; linker object is driven correctly.
;
; ------------------------------------------------------------------------------
;
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
@__oclc_finite_only_opt = external local_unnamed_addr addrspace(4) constant i8, align 1
@__oclc_ISA_version = external local_unnamed_addr addrspace(4) constant i32, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
declare float @__ocml_fmuladd_f32(float %0, float %1, float %2) #0

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define linkonce_odr hidden noundef float @__ocml_exp_f32(float noundef %0) local_unnamed_addr #0 {
  %2 = tail call float @llvm.exp.f32(float %0)
  ret float %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define linkonce_odr hidden double @__ocml_exp_f64(double noundef %0) local_unnamed_addr #0 {
  %2 = fmul double %0, 0x3FF71547652B82FE
  %3 = tail call double @llvm.rint.f64(double %2)
  %4 = fneg double %3
  %5 = tail call double @llvm.fma.f64(double %4, double 0x3FE62E42FEFA39EF, double %0)
  %6 = tail call double @llvm.fma.f64(double %4, double 0x3C7ABC9E3B39803F, double %5)
  %7 = tail call double @llvm.fma.f64(double %6, double 0x3E5ADE156A5DCB37, double 0x3E928AF3FCA7AB0C)
  %8 = tail call double @llvm.fma.f64(double %6, double %7, double 0x3EC71DEE623FDE64)
  %9 = tail call double @llvm.fma.f64(double %6, double %8, double 0x3EFA01997C89E6B0)
  %10 = tail call double @llvm.fma.f64(double %6, double %9, double 0x3F2A01A014761F6E)
  %11 = tail call double @llvm.fma.f64(double %6, double %10, double 0x3F56C16C1852B7B0)
  %12 = tail call double @llvm.fma.f64(double %6, double %11, double 0x3F81111111122322)
  %13 = tail call double @llvm.fma.f64(double %6, double %12, double 0x3FA55555555502A1)
  %14 = tail call double @llvm.fma.f64(double %6, double %13, double 0x3FC5555555555511)
  %15 = tail call double @llvm.fma.f64(double %6, double %14, double 0x3FE000000000000B)
  %16 = tail call double @llvm.fma.f64(double %6, double %15, double 1.000000e+00)
  %17 = tail call double @llvm.fma.f64(double %6, double %16, double 1.000000e+00)
  %18 = fptosi double %3 to i32
  %19 = tail call double @llvm.ldexp.f64.i32(double %17, i32 %18)
  %20 = load i8, ptr addrspace(4) @__oclc_finite_only_opt, align 1, !tbaa !4, !range !8, !noundef !9
  %21 = trunc nuw i8 %20 to i1
  %22 = fcmp ogt double %0, 1.024000e+03
  %23 = select i1 %22, double 0x7FF0000000000000, double %19
  %24 = select i1 %21, double %19, double %23
  %25 = fcmp olt double %0, -1.075000e+03
  %26 = select i1 %25, double 0.000000e+00, double %24
  ret double %26
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(none)
define linkonce_odr hidden double @__ocml_erfc_f64(double noundef %0) local_unnamed_addr #2 {
  %2 = tail call double @llvm.fabs.f64(double %0)
  %3 = fneg double %0
  %4 = fmul double %3, %0
  %5 = fneg double %4
  %6 = tail call double @llvm.fma.f64(double %3, double %0, double %5)
  %7 = tail call double @__ocml_exp_f64(double noundef %4) #15
  %8 = tail call double @llvm.fma.f64(double %7, double %6, double %7)
  %9 = tail call double @__ocmlpriv_erfcx_f64(double noundef %2) #15
  %10 = fmul double %8, %9
  %11 = fcmp ogt double %2, 0x403B39DC41E48BFC
  %12 = select i1 %11, double 0.000000e+00, double %10
  %13 = fsub double 2.000000e+00, %12
  %14 = fcmp olt double %0, 0.000000e+00
  %15 = select i1 %14, double %13, double %12
  ret double %15
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define linkonce_odr hidden double @__ocmlpriv_erfcx_f64(double noundef %0) local_unnamed_addr #0 {
  %2 = fadd double %0, -4.000000e+00
  %3 = fadd double %0, 4.000000e+00
  %4 = tail call double @llvm.amdgcn.rcp.f64(double %3)
  %5 = fneg double %3
  %6 = tail call double @llvm.fma.f64(double %5, double %4, double 1.000000e+00)
  %7 = tail call double @llvm.fma.f64(double %6, double %4, double %4)
  %8 = tail call double @llvm.fma.f64(double %5, double %7, double 1.000000e+00)
  %9 = tail call double @llvm.fma.f64(double %8, double %7, double %7)
  %10 = fmul double %2, %9
  %11 = fneg double %10
  %12 = fadd double %10, 1.000000e+00
  %13 = tail call double @llvm.fma.f64(double %12, double -4.000000e+00, double %0)
  %14 = tail call double @llvm.fma.f64(double %11, double %0, double %13)
  %15 = tail call double @llvm.fma.f64(double %9, double %14, double %10)
  %16 = tail call double @llvm.fma.f64(double %15, double 0xBE41F39D54DF3C0E, double 0xBE41166337CFA789)
  %17 = tail call double @llvm.fma.f64(double %15, double %16, double 0x3E7B45F1D9802B82)
  %18 = tail call double @llvm.fma.f64(double %15, double %17, double 0x3E6D90488A03DCDB)
  %19 = tail call double @llvm.fma.f64(double %15, double %18, double 0xBEAB87B02EBA62D8)
  %20 = tail call double @llvm.fma.f64(double %15, double %19, double 0x3E95104BA56E15F1)
  %21 = tail call double @llvm.fma.f64(double %15, double %20, double 0x3ED7F29F71C907DE)
  %22 = tail call double @llvm.fma.f64(double %15, double %21, double 0xBEE78F5C2CD770FB)
  %23 = tail call double @llvm.fma.f64(double %15, double %22, double 0xBEF995FB76D0A51A)
  %24 = tail call double @llvm.fma.f64(double %15, double %23, double 0x3F23BE2EC022D0ED)
  %25 = tail call double @llvm.fma.f64(double %15, double %24, double 0xBF2A1DEB2FDBF62E)
  %26 = tail call double @llvm.fma.f64(double %15, double %25, double 0xBF48D4AC3689FC43)
  %27 = tail call double @llvm.fma.f64(double %15, double %26, double 0x3F749C67192D909B)
  %28 = tail call double @llvm.fma.f64(double %15, double %27, double 0xBF909623852FF070)
  %29 = tail call double @llvm.fma.f64(double %15, double %28, double 0x3FA3079EDFADEA8F)
  %30 = tail call double @llvm.fma.f64(double %15, double %29, double 0xBFB0FB06DFF65910)
  %31 = tail call double @llvm.fma.f64(double %15, double %30, double 0x3FB7FEE004DE8F32)
  %32 = tail call double @llvm.fma.f64(double %15, double %31, double 0xBFB9DDB23C3DBEB3)
  %33 = tail call double @llvm.fma.f64(double %15, double %32, double 0x3FB16ECEFCFA6930)
  %34 = tail call double @llvm.fma.f64(double %15, double %33, double 0x3F8F7F5DF66FB8A3)
  %35 = tail call double @llvm.fma.f64(double %15, double %34, double 0xBFC1DF1AD154A2A8)
  %36 = tail call double @llvm.fma.f64(double %15, double %35, double 0x3FCDD2C8B74FEBF8)
  %37 = fadd double %0, %0
  %38 = fadd double %37, 1.000000e+00
  %39 = tail call double @llvm.amdgcn.rcp.f64(double %38)
  %40 = fneg double %38
  %41 = tail call double @llvm.fma.f64(double %40, double %39, double 1.000000e+00)
  %42 = tail call double @llvm.fma.f64(double %41, double %39, double %39)
  %43 = tail call double @llvm.fma.f64(double %40, double %42, double 1.000000e+00)
  %44 = tail call double @llvm.fma.f64(double %43, double %42, double %42)
  %45 = tail call double @llvm.fma.f64(double %36, double %44, double %44)
  %46 = fneg double %45
  %47 = tail call double @llvm.fma.f64(double %46, double %37, double 1.000000e+00)
  %48 = fsub double %36, %45
  %49 = fadd double %47, %48
  %50 = tail call double @llvm.fma.f64(double %44, double %49, double %45)
  ret double %50
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(none)
define linkonce_odr hidden float @__ocml_erfc_f32(float noundef %0) local_unnamed_addr #2 {
  %2 = tail call float @llvm.fabs.f32(float %0)
  %3 = fneg float %0
  %4 = fmul float %3, %0
  %5 = fneg float %4
  %6 = tail call float @llvm.fma.f32(float %3, float %0, float %5)
  %7 = tail call float @__ocml_exp_f32(float noundef %4) #15
  %8 = tail call float @llvm.fma.f32(float %7, float %6, float %7)
  %9 = tail call float @__ocmlpriv_erfcx_f32(float noundef %2) #15
  %10 = fmul float %8, %9
  %11 = fcmp ogt float %2, 0x40241BBF80000000
  %12 = select i1 %11, float 0.000000e+00, float %10
  %13 = fsub float 2.000000e+00, %12
  %14 = fcmp olt float %0, 0.000000e+00
  %15 = select i1 %14, float %13, float %12
  ret float %15
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(none)
define linkonce_odr hidden float @__ocmlpriv_erfcx_f32(float noundef %0) local_unnamed_addr #2 {
  %2 = fadd float %0, -2.000000e+00
  %3 = fadd float %0, 2.000000e+00
  %4 = tail call float @llvm.amdgcn.rcp.f32(float %3)
  %5 = fmul float %2, %4
  %6 = fneg float %5
  %7 = fadd float %5, 1.000000e+00
  %8 = tail call float @llvm.fma.f32(float %7, float -2.000000e+00, float %0)
  %9 = tail call float @llvm.fma.f32(float %6, float %0, float %8)
  %10 = tail call float @llvm.fma.f32(float %4, float %9, float %5)
  %11 = tail call float @__ocml_fmuladd_f32(float noundef %10, float noundef 0xBF3ADF1880000000, float noundef 0xBF545AEA60000000) #15
  %12 = tail call float @__ocml_fmuladd_f32(float noundef %10, float noundef %11, float noundef 0x3F55A5F680000000) #15
  %13 = tail call float @__ocml_fmuladd_f32(float noundef %10, float noundef %12, float noundef 0x3F81B44CE0000000) #15
  %14 = tail call float @__ocml_fmuladd_f32(float noundef %10, float noundef %13, float noundef 0xBF8082B620000000) #15
  %15 = tail call float @__ocml_fmuladd_f32(float noundef %10, float noundef %14, float noundef 0xBFABC14300000000) #15
  %16 = tail call float @__ocml_fmuladd_f32(float noundef %10, float noundef %15, float noundef 0x3FC4FFC540000000) #15
  %17 = tail call float @__ocml_fmuladd_f32(float noundef %10, float noundef %16, float noundef 0xBFC5407FA0000000) #15
  %18 = tail call float @__ocml_fmuladd_f32(float noundef %10, float noundef %17, float noundef 0xBFB7BF6160000000) #15
  %19 = tail call float @__ocml_fmuladd_f32(float noundef %10, float noundef %18, float noundef 0x3FD1BA0380000000) #15
  %20 = fadd float %0, %0
  %21 = fadd float %20, 1.000000e+00
  %22 = tail call float @llvm.amdgcn.rcp.f32(float %21)
  %23 = tail call float @llvm.fma.f32(float %19, float %22, float %22)
  %24 = fneg float %23
  %25 = tail call float @llvm.fma.f32(float %24, float %20, float 1.000000e+00)
  %26 = fsub float %19, %23
  %27 = fadd float %25, %26
  %28 = tail call float @llvm.fma.f32(float %22, float %27, float %23)
  ret float %28
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
