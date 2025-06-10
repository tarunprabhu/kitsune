; This is a very stripped down version of the actual libdevice bitcode file
; taken from some cuda implementation.

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-gpulibs"

%struct.uint2 = type { i32, i32 }

@.str = private unnamed_addr constant [11 x i8] c"__CUDA_FTZ\00", align 1
@__cudart_i2opi_f = internal addrspace(1) global [6 x i32] [i32 1011060801, i32 -614296167, i32 -181084736, i32 -64530479, i32 1313084713, i32 -1560706194], align 4

declare i32 @__nvvm_reflect(ptr) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fma.rn.ftz.f(float, float, float) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sin.approx.ftz.f(float) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sin.approx.f(float) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.cos.approx.ftz.f(float) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.cos.approx.f(float) #3

; Function Attrs: alwaysinline nounwind
define float @__nv_sinf(float %a) #0 {
  %result.i.i.i = alloca [7 x i32], align 4
  %1 = fmul float %a, 0x3FE45F3060000000
  %2 = call i32 @__nvvm_reflect(ptr @.str) #6
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %4, label %6

4:                                                ; preds = %0
  %5 = call i32 @llvm.nvvm.f2i.rn.ftz(float %1) #6
  br label %__nv_float2int_rn.exit.i.i

6:                                                ; preds = %0
  %7 = call i32 @llvm.nvvm.f2i.rn(float %1) #6
  br label %__nv_float2int_rn.exit.i.i

__nv_float2int_rn.exit.i.i:                       ; preds = %6, %4
  %.01 = phi i32 [ %5, %4 ], [ %7, %6 ]
  %8 = sitofp i32 %.01 to float
  %9 = call i32 @__nvvm_reflect(ptr @.str) #6
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %11, label %13

11:                                               ; preds = %__nv_float2int_rn.exit.i.i
  %12 = call float @llvm.nvvm.fma.rn.ftz.f(float %8, float 0xBFF921FB40000000, float %a) #6
  br label %__nv_fmaf_rn.exit.i.i

13:                                               ; preds = %__nv_float2int_rn.exit.i.i
  %14 = call float @llvm.nvvm.fma.rn.f(float %8, float 0xBFF921FB40000000, float %a) #6
  br label %__nv_fmaf_rn.exit.i.i

__nv_fmaf_rn.exit.i.i:                            ; preds = %13, %11
  %.02 = phi float [ %12, %11 ], [ %14, %13 ]
  %15 = call i32 @__nvvm_reflect(ptr @.str) #6
  %16 = icmp ne i32 %15, 0
  br i1 %16, label %17, label %19

17:                                               ; preds = %__nv_fmaf_rn.exit.i.i
  %18 = call float @llvm.nvvm.fma.rn.ftz.f(float %8, float 0xBE74442D00000000, float %.02) #6
  br label %__nv_fmaf_rn.exit1.i.i

19:                                               ; preds = %__nv_fmaf_rn.exit.i.i
  %20 = call float @llvm.nvvm.fma.rn.f(float %8, float 0xBE74442D00000000, float %.02) #6
  br label %__nv_fmaf_rn.exit1.i.i

__nv_fmaf_rn.exit1.i.i:                           ; preds = %19, %17
  %.03 = phi float [ %18, %17 ], [ %20, %19 ]
  %21 = call i32 @__nvvm_reflect(ptr @.str) #6
  %22 = icmp ne i32 %21, 0
  br i1 %22, label %23, label %25

23:                                               ; preds = %__nv_fmaf_rn.exit1.i.i
  %24 = call float @llvm.nvvm.fma.rn.ftz.f(float %8, float 0xBCF84698A0000000, float %.03) #6
  br label %__nv_fmaf_rn.exit2.i.i

25:                                               ; preds = %__nv_fmaf_rn.exit1.i.i
  %26 = call float @llvm.nvvm.fma.rn.f(float %8, float 0xBCF84698A0000000, float %.03) #6
  br label %__nv_fmaf_rn.exit2.i.i

__nv_fmaf_rn.exit2.i.i:                           ; preds = %25, %23
  %.04 = phi float [ %24, %23 ], [ %26, %25 ]
  %27 = call i32 @__nvvm_reflect(ptr @.str) #6
  %28 = icmp ne i32 %27, 0
  br i1 %28, label %29, label %31

29:                                               ; preds = %__nv_fmaf_rn.exit2.i.i
  %30 = call float @llvm.nvvm.fabs.ftz.f(float %a) #6
  br label %__nv_fabsf.exit.i.i

31:                                               ; preds = %__nv_fmaf_rn.exit2.i.i
  %32 = call float @llvm.nvvm.fabs.f(float %a) #6
  br label %__nv_fabsf.exit.i.i

__nv_fabsf.exit.i.i:                              ; preds = %31, %29
  %.06 = phi float [ %30, %29 ], [ %32, %31 ]
  %33 = fcmp oge float %.06, 1.056150e+05
  br i1 %33, label %34, label %__internal_trig_reduction_kernel.exit.i

34:                                               ; preds = %__nv_fabsf.exit.i.i
  %35 = call i32 @__nvvm_reflect(ptr @.str) #6
  %36 = icmp ne i32 %35, 0
  br i1 %36, label %37, label %39

37:                                               ; preds = %34
  %38 = call float @llvm.nvvm.fabs.ftz.f(float %a) #6
  br label %__nv_isinff.exit.i.i

39:                                               ; preds = %34
  %40 = call float @llvm.nvvm.fabs.f(float %a) #6
  br label %__nv_isinff.exit.i.i

__nv_isinff.exit.i.i:                             ; preds = %39, %37
  %.07 = phi float [ %38, %37 ], [ %40, %39 ]
  %41 = bitcast i32 2139095040 to float
  %42 = fcmp oeq float %.07, %41
  %43 = select i1 %42, i32 1, i32 0
  br i1 %42, label %44, label %51

44:                                               ; preds = %__nv_isinff.exit.i.i
  %45 = call i32 @__nvvm_reflect(ptr @.str) #6
  %46 = icmp ne i32 %45, 0
  br i1 %46, label %47, label %49

47:                                               ; preds = %44
  %48 = call float @llvm.nvvm.mul.rn.ftz.f(float %a, float 0.000000e+00) #6
  br label %__nv_fmul_rn.exit.i.i

49:                                               ; preds = %44
  %50 = call float @llvm.nvvm.mul.rn.f(float %a, float 0.000000e+00) #6
  br label %__nv_fmul_rn.exit.i.i

__nv_fmul_rn.exit.i.i:                            ; preds = %49, %47
  %.08 = phi float [ %48, %47 ], [ %50, %49 ]
  br label %127

51:                                               ; preds = %__nv_isinff.exit.i.i
  %52 = bitcast float %a to i32
  %53 = and i32 %52, -2147483648
  %54 = lshr i32 %52, 23
  %55 = and i32 %54, 255
  %56 = sub i32 %55, 128
  %57 = shl i32 %52, 8
  %58 = or i32 %57, -2147483648
  %59 = lshr i32 %56, 5
  %60 = sub i32 4, %59
  br label %61

61:                                               ; preds = %63, %51
  %hi.i.i.i.0 = phi i32 [ 0, %51 ], [ %71, %63 ]
  %iq.i.i.i.0 = phi i32 [ 0, %51 ], [ %74, %63 ]
  %62 = icmp slt i32 %iq.i.i.i.0, 6
  br i1 %62, label %63, label %75

63:                                               ; preds = %61
  %64 = sext i32 %iq.i.i.i.0 to i64
  %65 = getelementptr inbounds [6 x i32], ptr addrspace(1) @__cudart_i2opi_f, i64 0, i64 %64
  %66 = load i32, ptr addrspace(1) %65, align 4
  %67 = call { i32, i32 } asm "{\0A\09mad.lo.cc.u32   $0, $2, $3, $4;\0A\09madc.hi.u32     $1, $2, $3,  0;\0A\09}", "=r,=r,r,r,r"(i32 %66, i32 %58, i32 %hi.i.i.i.0) #7, !srcloc !12
  %68 = extractvalue { i32, i32 } %67, 0
  %69 = extractvalue { i32, i32 } %67, 1
  %insert = insertvalue %struct.uint2 undef, i32 %68, 0
  %insert25 = insertvalue %struct.uint2 %insert, i32 %69, 1
  %70 = extractvalue %struct.uint2 %insert25, 0
  %71 = extractvalue %struct.uint2 %insert25, 1
  %72 = sext i32 %iq.i.i.i.0 to i64
  %73 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %72
  store i32 %70, ptr %73, align 4
  %74 = add nsw i32 %iq.i.i.i.0, 1
  br label %61, !llvm.loop !13

75:                                               ; preds = %61
  %76 = sext i32 %iq.i.i.i.0 to i64
  %77 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %76
  store i32 %hi.i.i.i.0, ptr %77, align 4
  %78 = and i32 %56, 31
  %79 = add i32 %60, 2
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %80
  %82 = load i32, ptr %81, align 4
  %83 = add i32 %60, 1
  %84 = sext i32 %83 to i64
  %85 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %84
  %86 = load i32, ptr %85, align 4
  %87 = icmp ne i32 %78, 0
  br i1 %87, label %88, label %99

88:                                               ; preds = %75
  %89 = sub i32 32, %78
  %90 = shl i32 %82, %78
  %91 = lshr i32 %86, %89
  %92 = add i32 %90, %91
  %93 = shl i32 %86, %78
  %94 = sext i32 %60 to i64
  %95 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %94
  %96 = load i32, ptr %95, align 4
  %97 = lshr i32 %96, %89
  %98 = add i32 %93, %97
  br label %99

99:                                               ; preds = %88, %75
  %hi.i.i.i.1 = phi i32 [ %92, %88 ], [ %82, %75 ]
  %lo.i.i.i.0 = phi i32 [ %98, %88 ], [ %86, %75 ]
  %100 = lshr i32 %hi.i.i.i.1, 30
  %101 = shl i32 %hi.i.i.i.1, 2
  %102 = lshr i32 %lo.i.i.i.0, 30
  %103 = add i32 %101, %102
  %104 = shl i32 %lo.i.i.i.0, 2
  %105 = lshr i32 %103, 31
  %106 = add i32 %100, %105
  %107 = icmp ne i32 %53, 0
  br i1 %107, label %108, label %110

108:                                              ; preds = %99
  %109 = sub i32 0, %106
  br label %110

110:                                              ; preds = %108, %99
  %q.i.i.i.0 = phi i32 [ %109, %108 ], [ %106, %99 ]
  %111 = icmp ne i32 %105, 0
  br i1 %111, label %112, label %116

112:                                              ; preds = %110
  %113 = xor i32 %103, -1
  %114 = xor i32 %104, -1
  %115 = xor i32 %53, -2147483648
  br label %116

116:                                              ; preds = %112, %110
  %s.i.i.i.0 = phi i32 [ %115, %112 ], [ %53, %110 ]
  %hi.i.i.i.2 = phi i32 [ %113, %112 ], [ %103, %110 ]
  %lo.i.i.i.1 = phi i32 [ %114, %112 ], [ %104, %110 ]
  %117 = zext i32 %hi.i.i.i.2 to i64
  %118 = shl i64 %117, 32
  %119 = zext i32 %lo.i.i.i.1 to i64
  %120 = or i64 %118, %119
  %121 = sitofp i64 %120 to double
  %122 = fmul double %121, 0x3BF921FB54442D19
  %123 = fptrunc double %122 to float
  %124 = icmp ne i32 %s.i.i.i.0, 0
  br i1 %124, label %125, label %__internal_trig_reduction_slowpath.exit.i.i

125:                                              ; preds = %116
  %126 = fsub float -0.000000e+00, %123
  br label %__internal_trig_reduction_slowpath.exit.i.i

__internal_trig_reduction_slowpath.exit.i.i:      ; preds = %125, %116
  %r.i.i.i.0 = phi float [ %126, %125 ], [ %123, %116 ]
  br label %127

127:                                              ; preds = %__internal_trig_reduction_slowpath.exit.i.i, %__nv_fmul_rn.exit.i.i
  %i.i.0 = phi i32 [ 0, %__nv_fmul_rn.exit.i.i ], [ %q.i.i.i.0, %__internal_trig_reduction_slowpath.exit.i.i ]
  %t.i.i.0 = phi float [ %.08, %__nv_fmul_rn.exit.i.i ], [ %r.i.i.i.0, %__internal_trig_reduction_slowpath.exit.i.i ]
  br label %__internal_trig_reduction_kernel.exit.i

__internal_trig_reduction_kernel.exit.i:          ; preds = %127, %__nv_fabsf.exit.i.i
  %i.i.1 = phi i32 [ %i.i.0, %127 ], [ %.01, %__nv_fabsf.exit.i.i ]
  %t.i.i.1 = phi float [ %t.i.i.0, %127 ], [ %.04, %__nv_fabsf.exit.i.i ]
  %128 = call i32 @__nvvm_reflect(ptr @.str) #6
  %129 = icmp ne i32 %128, 0
  br i1 %129, label %130, label %132

130:                                              ; preds = %__internal_trig_reduction_kernel.exit.i
  %131 = call float @llvm.nvvm.mul.rn.ftz.f(float %t.i.i.1, float %t.i.i.1) #6
  br label %__nv_fmul_rn.exit.i2.i

132:                                              ; preds = %__internal_trig_reduction_kernel.exit.i
  %133 = call float @llvm.nvvm.mul.rn.f(float %t.i.i.1, float %t.i.i.1) #6
  br label %__nv_fmul_rn.exit.i2.i

__nv_fmul_rn.exit.i2.i:                           ; preds = %132, %130
  %.011 = phi float [ %131, %130 ], [ %133, %132 ]
  %134 = and i32 %i.i.1, 1
  %135 = icmp ne i32 %134, 0
  br i1 %135, label %136, label %137

136:                                              ; preds = %__nv_fmul_rn.exit.i2.i
  br label %138

137:                                              ; preds = %__nv_fmul_rn.exit.i2.i
  br label %138

138:                                              ; preds = %137, %136
  %139 = phi float [ 1.000000e+00, %136 ], [ %t.i.i.1, %137 ]
  %140 = call i32 @__nvvm_reflect(ptr @.str) #6
  %141 = icmp ne i32 %140, 0
  br i1 %141, label %142, label %144

142:                                              ; preds = %138
  %143 = call float @llvm.nvvm.fma.rn.ftz.f(float %.011, float %139, float 0.000000e+00) #6
  br label %__internal_fmad.exit.i.i

144:                                              ; preds = %138
  %145 = call float @llvm.nvvm.fma.rn.f(float %.011, float %139, float 0.000000e+00) #6
  br label %__internal_fmad.exit.i.i

__internal_fmad.exit.i.i:                         ; preds = %144, %142
  %.012 = phi float [ %143, %142 ], [ %145, %144 ]
  %146 = and i32 %i.i.1, 1
  %147 = icmp ne i32 %146, 0
  br i1 %147, label %148, label %155

148:                                              ; preds = %__internal_fmad.exit.i.i
  %149 = call i32 @__nvvm_reflect(ptr @.str) #6
  %150 = icmp ne i32 %149, 0
  br i1 %150, label %151, label %153

151:                                              ; preds = %148
  %152 = call float @llvm.nvvm.fma.rn.ftz.f(float 0x3EF9758000000000, float %.011, float 0xBF56C0FDA0000000) #6
  br label %__internal_fmad.exit1.i.i

153:                                              ; preds = %148
  %154 = call float @llvm.nvvm.fma.rn.f(float 0x3EF9758000000000, float %.011, float 0xBF56C0FDA0000000) #6
  br label %__internal_fmad.exit1.i.i

__internal_fmad.exit1.i.i:                        ; preds = %153, %151
  %.013 = phi float [ %152, %151 ], [ %154, %153 ]
  br label %156

155:                                              ; preds = %__internal_fmad.exit.i.i
  br label %156

156:                                              ; preds = %155, %__internal_fmad.exit1.i.i
  %157 = phi float [ %.013, %__internal_fmad.exit1.i.i ], [ 0xBF29A82A60000000, %155 ]
  %158 = and i32 %i.i.1, 1
  %159 = icmp ne i32 %158, 0
  %160 = select i1 %159, float 0x3FA5555760000000, float 0x3F8110BC80000000
  %161 = call i32 @__nvvm_reflect(ptr @.str) #6
  %162 = icmp ne i32 %161, 0
  br i1 %162, label %163, label %165

163:                                              ; preds = %156
  %164 = call float @llvm.nvvm.fma.rn.ftz.f(float %157, float %.011, float %160) #6
  br label %__internal_fmad.exit2.i.i

165:                                              ; preds = %156
  %166 = call float @llvm.nvvm.fma.rn.f(float %157, float %.011, float %160) #6
  br label %__internal_fmad.exit2.i.i

__internal_fmad.exit2.i.i:                        ; preds = %165, %163
  %.010 = phi float [ %164, %163 ], [ %166, %165 ]
  %167 = and i32 %i.i.1, 1
  %168 = icmp ne i32 %167, 0
  %169 = select i1 %168, float 0xBFDFFFFFE0000000, float 0xBFC5555500000000
  %170 = call i32 @__nvvm_reflect(ptr @.str) #6
  %171 = icmp ne i32 %170, 0
  br i1 %171, label %172, label %174

172:                                              ; preds = %__internal_fmad.exit2.i.i
  %173 = call float @llvm.nvvm.fma.rn.ftz.f(float %.010, float %.011, float %169) #6
  br label %__internal_fmad.exit3.i.i

174:                                              ; preds = %__internal_fmad.exit2.i.i
  %175 = call float @llvm.nvvm.fma.rn.f(float %.010, float %.011, float %169) #6
  br label %__internal_fmad.exit3.i.i

__internal_fmad.exit3.i.i:                        ; preds = %174, %172
  %.09 = phi float [ %173, %172 ], [ %175, %174 ]
  %176 = call i32 @__nvvm_reflect(ptr @.str) #6
  %177 = icmp ne i32 %176, 0
  br i1 %177, label %178, label %180

178:                                              ; preds = %__internal_fmad.exit3.i.i
  %179 = call float @llvm.nvvm.fma.rn.ftz.f(float %.09, float %.012, float %139) #6
  br label %__internal_fmad.exit4.i.i

180:                                              ; preds = %__internal_fmad.exit3.i.i
  %181 = call float @llvm.nvvm.fma.rn.f(float %.09, float %.012, float %139) #6
  br label %__internal_fmad.exit4.i.i

__internal_fmad.exit4.i.i:                        ; preds = %180, %178
  %.05 = phi float [ %179, %178 ], [ %181, %180 ]
  %182 = and i32 %i.i.1, 2
  %183 = icmp ne i32 %182, 0
  br i1 %183, label %184, label %__internal_accurate_sinf.exit

184:                                              ; preds = %__internal_fmad.exit4.i.i
  %185 = call i32 @__nvvm_reflect(ptr @.str) #6
  %186 = icmp ne i32 %185, 0
  br i1 %186, label %187, label %189

187:                                              ; preds = %184
  %188 = call float @llvm.nvvm.fma.rn.ftz.f(float %.05, float -1.000000e+00, float 0.000000e+00) #6
  br label %__internal_fmad.exit5.i.i

189:                                              ; preds = %184
  %190 = call float @llvm.nvvm.fma.rn.f(float %.05, float -1.000000e+00, float 0.000000e+00) #6
  br label %__internal_fmad.exit5.i.i

__internal_fmad.exit5.i.i:                        ; preds = %189, %187
  %.0 = phi float [ %188, %187 ], [ %190, %189 ]
  br label %__internal_accurate_sinf.exit

__internal_accurate_sinf.exit:                    ; preds = %__internal_fmad.exit4.i.i, %__internal_fmad.exit5.i.i
  %z.i.i.0 = phi float [ %.0, %__internal_fmad.exit5.i.i ], [ %.05, %__internal_fmad.exit4.i.i ]
  ret float %z.i.i.0
}

; Function Attrs: alwaysinline nounwind
define float @__nv_fast_sinf(float %a) #0 {
  %1 = call i32 @__nvvm_reflect(ptr @.str) #6
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %3, label %5

3:                                                ; preds = %0
  %4 = call float @llvm.nvvm.sin.approx.ftz.f(float %a) #6
  br label %__nvvm_builtin_sinf.exit

5:                                                ; preds = %0
  %6 = call float @llvm.nvvm.sin.approx.f(float %a) #6
  br label %__nvvm_builtin_sinf.exit

__nvvm_builtin_sinf.exit:                         ; preds = %3, %5
  %.0 = phi float [ %4, %3 ], [ %6, %5 ]
  ret float %.0
}

; Function Attrs: alwaysinline nounwind
define float @__nv_cosf(float %a) #0 {
  %result.i.i.i = alloca [7 x i32], align 4
  %1 = fmul float %a, 0x3FE45F3060000000
  %2 = call i32 @__nvvm_reflect(ptr @.str) #6
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %4, label %6

4:                                                ; preds = %0
  %5 = call i32 @llvm.nvvm.f2i.rn.ftz(float %1) #6
  br label %__nv_float2int_rn.exit.i.i

6:                                                ; preds = %0
  %7 = call i32 @llvm.nvvm.f2i.rn(float %1) #6
  br label %__nv_float2int_rn.exit.i.i

__nv_float2int_rn.exit.i.i:                       ; preds = %6, %4
  %.01 = phi i32 [ %5, %4 ], [ %7, %6 ]
  %8 = sitofp i32 %.01 to float
  %9 = call i32 @__nvvm_reflect(ptr @.str) #6
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %11, label %13

11:                                               ; preds = %__nv_float2int_rn.exit.i.i
  %12 = call float @llvm.nvvm.fma.rn.ftz.f(float %8, float 0xBFF921FB40000000, float %a) #6
  br label %__nv_fmaf_rn.exit.i.i

13:                                               ; preds = %__nv_float2int_rn.exit.i.i
  %14 = call float @llvm.nvvm.fma.rn.f(float %8, float 0xBFF921FB40000000, float %a) #6
  br label %__nv_fmaf_rn.exit.i.i

__nv_fmaf_rn.exit.i.i:                            ; preds = %13, %11
  %.02 = phi float [ %12, %11 ], [ %14, %13 ]
  %15 = call i32 @__nvvm_reflect(ptr @.str) #6
  %16 = icmp ne i32 %15, 0
  br i1 %16, label %17, label %19

17:                                               ; preds = %__nv_fmaf_rn.exit.i.i
  %18 = call float @llvm.nvvm.fma.rn.ftz.f(float %8, float 0xBE74442D00000000, float %.02) #6
  br label %__nv_fmaf_rn.exit1.i.i

19:                                               ; preds = %__nv_fmaf_rn.exit.i.i
  %20 = call float @llvm.nvvm.fma.rn.f(float %8, float 0xBE74442D00000000, float %.02) #6
  br label %__nv_fmaf_rn.exit1.i.i

__nv_fmaf_rn.exit1.i.i:                           ; preds = %19, %17
  %.03 = phi float [ %18, %17 ], [ %20, %19 ]
  %21 = call i32 @__nvvm_reflect(ptr @.str) #6
  %22 = icmp ne i32 %21, 0
  br i1 %22, label %23, label %25

23:                                               ; preds = %__nv_fmaf_rn.exit1.i.i
  %24 = call float @llvm.nvvm.fma.rn.ftz.f(float %8, float 0xBCF84698A0000000, float %.03) #6
  br label %__nv_fmaf_rn.exit2.i.i

25:                                               ; preds = %__nv_fmaf_rn.exit1.i.i
  %26 = call float @llvm.nvvm.fma.rn.f(float %8, float 0xBCF84698A0000000, float %.03) #6
  br label %__nv_fmaf_rn.exit2.i.i

__nv_fmaf_rn.exit2.i.i:                           ; preds = %25, %23
  %.04 = phi float [ %24, %23 ], [ %26, %25 ]
  %27 = call i32 @__nvvm_reflect(ptr @.str) #6
  %28 = icmp ne i32 %27, 0
  br i1 %28, label %29, label %31

29:                                               ; preds = %__nv_fmaf_rn.exit2.i.i
  %30 = call float @llvm.nvvm.fabs.ftz.f(float %a) #6
  br label %__nv_fabsf.exit.i.i

31:                                               ; preds = %__nv_fmaf_rn.exit2.i.i
  %32 = call float @llvm.nvvm.fabs.f(float %a) #6
  br label %__nv_fabsf.exit.i.i

__nv_fabsf.exit.i.i:                              ; preds = %31, %29
  %.06 = phi float [ %30, %29 ], [ %32, %31 ]
  %33 = fcmp oge float %.06, 1.056150e+05
  br i1 %33, label %34, label %__internal_trig_reduction_kernel.exit.i

34:                                               ; preds = %__nv_fabsf.exit.i.i
  %35 = call i32 @__nvvm_reflect(ptr @.str) #6
  %36 = icmp ne i32 %35, 0
  br i1 %36, label %37, label %39

37:                                               ; preds = %34
  %38 = call float @llvm.nvvm.fabs.ftz.f(float %a) #6
  br label %__nv_isinff.exit.i.i

39:                                               ; preds = %34
  %40 = call float @llvm.nvvm.fabs.f(float %a) #6
  br label %__nv_isinff.exit.i.i

__nv_isinff.exit.i.i:                             ; preds = %39, %37
  %.07 = phi float [ %38, %37 ], [ %40, %39 ]
  %41 = bitcast i32 2139095040 to float
  %42 = fcmp oeq float %.07, %41
  %43 = select i1 %42, i32 1, i32 0
  br i1 %42, label %44, label %51

44:                                               ; preds = %__nv_isinff.exit.i.i
  %45 = call i32 @__nvvm_reflect(ptr @.str) #6
  %46 = icmp ne i32 %45, 0
  br i1 %46, label %47, label %49

47:                                               ; preds = %44
  %48 = call float @llvm.nvvm.mul.rn.ftz.f(float %a, float 0.000000e+00) #6
  br label %__nv_fmul_rn.exit.i.i

49:                                               ; preds = %44
  %50 = call float @llvm.nvvm.mul.rn.f(float %a, float 0.000000e+00) #6
  br label %__nv_fmul_rn.exit.i.i

__nv_fmul_rn.exit.i.i:                            ; preds = %49, %47
  %.08 = phi float [ %48, %47 ], [ %50, %49 ]
  br label %127

51:                                               ; preds = %__nv_isinff.exit.i.i
  %52 = bitcast float %a to i32
  %53 = and i32 %52, -2147483648
  %54 = lshr i32 %52, 23
  %55 = and i32 %54, 255
  %56 = sub i32 %55, 128
  %57 = shl i32 %52, 8
  %58 = or i32 %57, -2147483648
  %59 = lshr i32 %56, 5
  %60 = sub i32 4, %59
  br label %61

61:                                               ; preds = %63, %51
  %hi.i.i.i.0 = phi i32 [ 0, %51 ], [ %71, %63 ]
  %iq.i.i.i.0 = phi i32 [ 0, %51 ], [ %74, %63 ]
  %62 = icmp slt i32 %iq.i.i.i.0, 6
  br i1 %62, label %63, label %75

63:                                               ; preds = %61
  %64 = sext i32 %iq.i.i.i.0 to i64
  %65 = getelementptr inbounds [6 x i32], ptr addrspace(1) @__cudart_i2opi_f, i64 0, i64 %64
  %66 = load i32, ptr addrspace(1) %65, align 4
  %67 = call { i32, i32 } asm "{\0A\09mad.lo.cc.u32   $0, $2, $3, $4;\0A\09madc.hi.u32     $1, $2, $3,  0;\0A\09}", "=r,=r,r,r,r"(i32 %66, i32 %58, i32 %hi.i.i.i.0) #7, !srcloc !12
  %68 = extractvalue { i32, i32 } %67, 0
  %69 = extractvalue { i32, i32 } %67, 1
  %insert = insertvalue %struct.uint2 undef, i32 %68, 0
  %insert25 = insertvalue %struct.uint2 %insert, i32 %69, 1
  %70 = extractvalue %struct.uint2 %insert25, 0
  %71 = extractvalue %struct.uint2 %insert25, 1
  %72 = sext i32 %iq.i.i.i.0 to i64
  %73 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %72
  store i32 %70, ptr %73, align 4
  %74 = add nsw i32 %iq.i.i.i.0, 1
  br label %61, !llvm.loop !13

75:                                               ; preds = %61
  %76 = sext i32 %iq.i.i.i.0 to i64
  %77 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %76
  store i32 %hi.i.i.i.0, ptr %77, align 4
  %78 = and i32 %56, 31
  %79 = add i32 %60, 2
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %80
  %82 = load i32, ptr %81, align 4
  %83 = add i32 %60, 1
  %84 = sext i32 %83 to i64
  %85 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %84
  %86 = load i32, ptr %85, align 4
  %87 = icmp ne i32 %78, 0
  br i1 %87, label %88, label %99

88:                                               ; preds = %75
  %89 = sub i32 32, %78
  %90 = shl i32 %82, %78
  %91 = lshr i32 %86, %89
  %92 = add i32 %90, %91
  %93 = shl i32 %86, %78
  %94 = sext i32 %60 to i64
  %95 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %94
  %96 = load i32, ptr %95, align 4
  %97 = lshr i32 %96, %89
  %98 = add i32 %93, %97
  br label %99

99:                                               ; preds = %88, %75
  %hi.i.i.i.1 = phi i32 [ %92, %88 ], [ %82, %75 ]
  %lo.i.i.i.0 = phi i32 [ %98, %88 ], [ %86, %75 ]
  %100 = lshr i32 %hi.i.i.i.1, 30
  %101 = shl i32 %hi.i.i.i.1, 2
  %102 = lshr i32 %lo.i.i.i.0, 30
  %103 = add i32 %101, %102
  %104 = shl i32 %lo.i.i.i.0, 2
  %105 = lshr i32 %103, 31
  %106 = add i32 %100, %105
  %107 = icmp ne i32 %53, 0
  br i1 %107, label %108, label %110

108:                                              ; preds = %99
  %109 = sub i32 0, %106
  br label %110

110:                                              ; preds = %108, %99
  %q.i.i.i.0 = phi i32 [ %109, %108 ], [ %106, %99 ]
  %111 = icmp ne i32 %105, 0
  br i1 %111, label %112, label %116

112:                                              ; preds = %110
  %113 = xor i32 %103, -1
  %114 = xor i32 %104, -1
  %115 = xor i32 %53, -2147483648
  br label %116

116:                                              ; preds = %112, %110
  %s.i.i.i.0 = phi i32 [ %115, %112 ], [ %53, %110 ]
  %hi.i.i.i.2 = phi i32 [ %113, %112 ], [ %103, %110 ]
  %lo.i.i.i.1 = phi i32 [ %114, %112 ], [ %104, %110 ]
  %117 = zext i32 %hi.i.i.i.2 to i64
  %118 = shl i64 %117, 32
  %119 = zext i32 %lo.i.i.i.1 to i64
  %120 = or i64 %118, %119
  %121 = sitofp i64 %120 to double
  %122 = fmul double %121, 0x3BF921FB54442D19
  %123 = fptrunc double %122 to float
  %124 = icmp ne i32 %s.i.i.i.0, 0
  br i1 %124, label %125, label %__internal_trig_reduction_slowpath.exit.i.i

125:                                              ; preds = %116
  %126 = fsub float -0.000000e+00, %123
  br label %__internal_trig_reduction_slowpath.exit.i.i

__internal_trig_reduction_slowpath.exit.i.i:      ; preds = %125, %116
  %r.i.i.i.0 = phi float [ %126, %125 ], [ %123, %116 ]
  br label %127

127:                                              ; preds = %__internal_trig_reduction_slowpath.exit.i.i, %__nv_fmul_rn.exit.i.i
  %i.i.0 = phi i32 [ 0, %__nv_fmul_rn.exit.i.i ], [ %q.i.i.i.0, %__internal_trig_reduction_slowpath.exit.i.i ]
  %t.i.i.0 = phi float [ %.08, %__nv_fmul_rn.exit.i.i ], [ %r.i.i.i.0, %__internal_trig_reduction_slowpath.exit.i.i ]
  br label %__internal_trig_reduction_kernel.exit.i

__internal_trig_reduction_kernel.exit.i:          ; preds = %127, %__nv_fabsf.exit.i.i
  %i.i.1 = phi i32 [ %i.i.0, %127 ], [ %.01, %__nv_fabsf.exit.i.i ]
  %t.i.i.1 = phi float [ %t.i.i.0, %127 ], [ %.04, %__nv_fabsf.exit.i.i ]
  %128 = add i32 %i.i.1, 1
  %129 = call i32 @__nvvm_reflect(ptr @.str) #6
  %130 = icmp ne i32 %129, 0
  br i1 %130, label %131, label %133

131:                                              ; preds = %__internal_trig_reduction_kernel.exit.i
  %132 = call float @llvm.nvvm.mul.rn.ftz.f(float %t.i.i.1, float %t.i.i.1) #6
  br label %__nv_fmul_rn.exit.i2.i

133:                                              ; preds = %__internal_trig_reduction_kernel.exit.i
  %134 = call float @llvm.nvvm.mul.rn.f(float %t.i.i.1, float %t.i.i.1) #6
  br label %__nv_fmul_rn.exit.i2.i

__nv_fmul_rn.exit.i2.i:                           ; preds = %133, %131
  %.011 = phi float [ %132, %131 ], [ %134, %133 ]
  %135 = and i32 %128, 1
  %136 = icmp ne i32 %135, 0
  br i1 %136, label %137, label %138

137:                                              ; preds = %__nv_fmul_rn.exit.i2.i
  br label %139

138:                                              ; preds = %__nv_fmul_rn.exit.i2.i
  br label %139

139:                                              ; preds = %138, %137
  %140 = phi float [ 1.000000e+00, %137 ], [ %t.i.i.1, %138 ]
  %141 = call i32 @__nvvm_reflect(ptr @.str) #6
  %142 = icmp ne i32 %141, 0
  br i1 %142, label %143, label %145

143:                                              ; preds = %139
  %144 = call float @llvm.nvvm.fma.rn.ftz.f(float %.011, float %140, float 0.000000e+00) #6
  br label %__internal_fmad.exit.i.i

145:                                              ; preds = %139
  %146 = call float @llvm.nvvm.fma.rn.f(float %.011, float %140, float 0.000000e+00) #6
  br label %__internal_fmad.exit.i.i

__internal_fmad.exit.i.i:                         ; preds = %145, %143
  %.012 = phi float [ %144, %143 ], [ %146, %145 ]
  %147 = and i32 %128, 1
  %148 = icmp ne i32 %147, 0
  br i1 %148, label %149, label %156

149:                                              ; preds = %__internal_fmad.exit.i.i
  %150 = call i32 @__nvvm_reflect(ptr @.str) #6
  %151 = icmp ne i32 %150, 0
  br i1 %151, label %152, label %154

152:                                              ; preds = %149
  %153 = call float @llvm.nvvm.fma.rn.ftz.f(float 0x3EF9758000000000, float %.011, float 0xBF56C0FDA0000000) #6
  br label %__internal_fmad.exit1.i.i

154:                                              ; preds = %149
  %155 = call float @llvm.nvvm.fma.rn.f(float 0x3EF9758000000000, float %.011, float 0xBF56C0FDA0000000) #6
  br label %__internal_fmad.exit1.i.i

__internal_fmad.exit1.i.i:                        ; preds = %154, %152
  %.013 = phi float [ %153, %152 ], [ %155, %154 ]
  br label %157

156:                                              ; preds = %__internal_fmad.exit.i.i
  br label %157

157:                                              ; preds = %156, %__internal_fmad.exit1.i.i
  %158 = phi float [ %.013, %__internal_fmad.exit1.i.i ], [ 0xBF29A82A60000000, %156 ]
  %159 = and i32 %128, 1
  %160 = icmp ne i32 %159, 0
  %161 = select i1 %160, float 0x3FA5555760000000, float 0x3F8110BC80000000
  %162 = call i32 @__nvvm_reflect(ptr @.str) #6
  %163 = icmp ne i32 %162, 0
  br i1 %163, label %164, label %166

164:                                              ; preds = %157
  %165 = call float @llvm.nvvm.fma.rn.ftz.f(float %158, float %.011, float %161) #6
  br label %__internal_fmad.exit2.i.i

166:                                              ; preds = %157
  %167 = call float @llvm.nvvm.fma.rn.f(float %158, float %.011, float %161) #6
  br label %__internal_fmad.exit2.i.i

__internal_fmad.exit2.i.i:                        ; preds = %166, %164
  %.010 = phi float [ %165, %164 ], [ %167, %166 ]
  %168 = and i32 %128, 1
  %169 = icmp ne i32 %168, 0
  %170 = select i1 %169, float 0xBFDFFFFFE0000000, float 0xBFC5555500000000
  %171 = call i32 @__nvvm_reflect(ptr @.str) #6
  %172 = icmp ne i32 %171, 0
  br i1 %172, label %173, label %175

173:                                              ; preds = %__internal_fmad.exit2.i.i
  %174 = call float @llvm.nvvm.fma.rn.ftz.f(float %.010, float %.011, float %170) #6
  br label %__internal_fmad.exit3.i.i

175:                                              ; preds = %__internal_fmad.exit2.i.i
  %176 = call float @llvm.nvvm.fma.rn.f(float %.010, float %.011, float %170) #6
  br label %__internal_fmad.exit3.i.i

__internal_fmad.exit3.i.i:                        ; preds = %175, %173
  %.09 = phi float [ %174, %173 ], [ %176, %175 ]
  %177 = call i32 @__nvvm_reflect(ptr @.str) #6
  %178 = icmp ne i32 %177, 0
  br i1 %178, label %179, label %181

179:                                              ; preds = %__internal_fmad.exit3.i.i
  %180 = call float @llvm.nvvm.fma.rn.ftz.f(float %.09, float %.012, float %140) #6
  br label %__internal_fmad.exit4.i.i

181:                                              ; preds = %__internal_fmad.exit3.i.i
  %182 = call float @llvm.nvvm.fma.rn.f(float %.09, float %.012, float %140) #6
  br label %__internal_fmad.exit4.i.i

__internal_fmad.exit4.i.i:                        ; preds = %181, %179
  %.05 = phi float [ %180, %179 ], [ %182, %181 ]
  %183 = and i32 %128, 2
  %184 = icmp ne i32 %183, 0
  br i1 %184, label %185, label %__internal_accurate_cosf.exit

185:                                              ; preds = %__internal_fmad.exit4.i.i
  %186 = call i32 @__nvvm_reflect(ptr @.str) #6
  %187 = icmp ne i32 %186, 0
  br i1 %187, label %188, label %190

188:                                              ; preds = %185
  %189 = call float @llvm.nvvm.fma.rn.ftz.f(float %.05, float -1.000000e+00, float 0.000000e+00) #6
  br label %__internal_fmad.exit5.i.i

190:                                              ; preds = %185
  %191 = call float @llvm.nvvm.fma.rn.f(float %.05, float -1.000000e+00, float 0.000000e+00) #6
  br label %__internal_fmad.exit5.i.i

__internal_fmad.exit5.i.i:                        ; preds = %190, %188
  %.0 = phi float [ %189, %188 ], [ %191, %190 ]
  br label %__internal_accurate_cosf.exit

__internal_accurate_cosf.exit:                    ; preds = %__internal_fmad.exit4.i.i, %__internal_fmad.exit5.i.i
  %z.i.i.0 = phi float [ %.0, %__internal_fmad.exit5.i.i ], [ %.05, %__internal_fmad.exit4.i.i ]
  ret float %z.i.i.0
}

; Function Attrs: alwaysinline nounwind
define float @__nv_fast_cosf(float %a) #0 {
  %1 = call i32 @__nvvm_reflect(ptr @.str) #6
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %3, label %5

3:                                                ; preds = %0
  %4 = call float @llvm.nvvm.cos.approx.ftz.f(float %a) #6
  br label %__nvvm_builtin_cosf.exit

5:                                                ; preds = %0
  %6 = call float @llvm.nvvm.cos.approx.f(float %a) #6
  br label %__nvvm_builtin_cosf.exit

__nvvm_builtin_cosf.exit:                         ; preds = %3, %5
  %.0 = phi float [ %4, %3 ], [ %6, %5 ]
  ret float %.0
}

attributes #0 = { alwaysinline nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #4 = { noinline nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { nounwind }
attributes #7 = { nounwind memory(none) }

!0 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!1 = !{i32 2, i32 0}
!2 = !{i32 21684}
!3 = !{i32 9503}
!4 = !{i32 9580}
!5 = !{i32 9939}
!6 = !{i32 10016}
!7 = !{i32 10375}
!8 = !{i32 10452}
!9 = !{i32 10811}
!10 = !{i32 10888}
!11 = !{i32 12369}
!12 = !{i32 33516, i32 33520, i32 33565, i32 33610}
!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.unroll.count", i32 1}
!15 = !{i32 157615, i32 157619, i32 157688, i32 157736, i32 157784, i32 157832, i32 157880, i32 157928, i32 157976, i32 158024, i32 158072, i32 158120, i32 158168, i32 158216, i32 158264, i32 158312, i32 158360}
!16 = distinct !{!16, !14}
!17 = !{i32 155349, i32 155353, i32 155424, i32 155466, i32 155508, i32 155550, i32 155592, i32 155634, i32 155676, i32 155718, i32 155760, i32 155802, i32 155844}
!18 = !{i32 156390, i32 156394, i32 156453, i32 156500, i32 156547, i32 156594, i32 156641, i32 156688, i32 156735, i32 156782, i32 156829, i32 156876, i32 156923, i32 156970, i32 157017, i32 157064}
!19 = !{i32 154151, i32 154155, i32 154226, i32 154268, i32 154310, i32 154352, i32 154394, i32 154436, i32 154478, i32 154520, i32 154562, i32 154604, i32 154646}
!20 = !{i32 151545}
!21 = !{i32 151166}
!22 = !{i32 287270}
!23 = !{i32 287497}
!24 = !{i32 287724}
!25 = !{i32 287951}
