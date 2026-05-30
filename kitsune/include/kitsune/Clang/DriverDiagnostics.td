//==- DriverDiagnostics.td - Driver diagnostics for Kitsune -*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kitsune-specific driver diagnostics.
//
// This file is intended to be included in clang/Basic/DiagnosticDriverKinds.td.
// It references entities defined in that file, or those included transitively
// from it. This should not, generally, be used directly.
// DiagnosticDriverKinds.td should be used instead.
//
//===----------------------------------------------------------------------===//

def warn_drv_kitsune_no_gpu : Warning<
  "%0. Using default '%1'">,
  InGroup<Kitsune>;
def warn_drv_kitsune_multi_gpu : Warning<
  "multiple %0 GPU architectures detected: '%1'">,
  InGroup<Kitsune>;

def err_drv_tapir_target_system: Error<
  "'%0' tapir target does not support system '%1'">;
def err_drv_tapir_target_arch: Error<
  "'%0' tapir target does not support architecture '%1'">;

def warn_drv_kitsune_cuda_optzns_debug : Warning<
  "ptxas does not support optimized debugging. Setting ptxas optimization "
  "level to -O0. All other optimizations will be performed at level '-O%0'">,
  InGroup<Kitsune>;

def err_drv_kitsune_cuda_optzns_debug : Error<
  "ptxas does not support optimized debugging. Use -O1 for partial debugging "
  "support">;

def err_drv_unsupported_option_argument_for_frontend : Error<
  "unsupported argument '%1' to option '%0' for frontend '%2'">;

def err_drv_kitsune_missing_required : Error<
  "missing required option '--%0'">;

def err_drv_kitsune_tapir_required : Error<
  "--tapir is required with '%0'">;

def err_drv_kitsune_plugin_missing : Error<
  "'--tapir-plugin' is required with the 'custom' tapir target">;
def err_drv_kitsune_plugin_multiple : Error<
  "'--tapir-plugin' must have a unique value">;
def err_drv_kitsune_plugin_wrong_target : Error<
  "'--tapir-plugin' requires '--tapir=custom'">;
def err_drv_kitsune_plugin_invalid : Error<
  "invalid tapir target plugin: %0">;

def err_drv_kitsune_optzns_required : Error<
  "--tapir requires optimization level O1 or higher">;
def err_drv_kitsune_lto_o2_required : Error<
  "--tapir requires optimization level O2 or higher for LTO">;
def err_drv_kitsune_bad_opt_level : Error<
  "unsupported optimization level '%0'">;

def err_drv_kitsune_threads_per_block : Error<
  "value of '%0' not in range [1,1024]">;

def err_drv_kitsune_bad_cuda_arch : Error<
  "unsupported NVIDIA GPU architecture '%0'">;

def err_drv_kitsune_bad_hip_arch : Error<
  "unsupported AMD GPU architecture '%0'">;
def err_drv_kitsune_hip_code_object_version : Error<
  "rocm code object version '%0' is not supported">;

def err_drv_kitsune_lto_disallowed_arg : Error<
  "'%0' cannot be used with -flto in Kitsune">;

def err_drv_kitsune_objc : Error<
  "kitsune does not support Objective-C">;
def err_drv_kitsune_opencl: Error<
  "kitsune does not support OpenCL">;
def err_drv_kitsune_cuda: Error<
  "kitsune does not support the Cuda language">;
def err_drv_kitsune_hip: Error<
  "kitsune does not support the Hip language">;
def err_drv_kitsune_offload : Error<
  "cannot use offload with a tapir target">;

def err_drv_opencilk_missing_abi_bitcode: Error<
  "cannot find OpenCilk runtime ABI bitcode file">;

def err_drv_kitsune_kokkos_disabled : Error<
  "kokkos support was not enabled when kitsune was built">;

def err_drv_kitsune_target_not_enabled: Error<
  "tapir target '%0' was not enabled when kitsune was built">;

def err_drv_kitsune_frontend_only: Error<
  "option '%0' must be used with a Kitsune frontend">;
