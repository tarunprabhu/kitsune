//===- DriverOptions.td - Options for Kitsune's frontends --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnostics emitted by Kitsune's C/C++ frontends, typically during "codegen",
// i.e. LLVM-IR generation.
//
// This file is intended to be included in clang/Driver/Options.td. It
// references entities defined in that file, or those included transitively from
// it. This should not, generally, be used directly. clang/Driver/Options.td
// should be used instead.
//
// Some considerations to keep in mind when adding new options here:
//
//   - The RenderStyle for any new options that are added to this must be
//     "Joined"
//
//   - A double-dash before the options should be preferred instead of a single
//      dash. This is for consistency with other tools such as opt and lld
//
//   - Avoid prefixing options with -f unless the intention is to make the
//     option available to "vanilla" clang/flang
//
//   - New options should be added to the kitsune_Group defined below. This will
//     ensure that a Kitsune frontend (kitcc, kit++, kitfc etc.) is used when
//     compiling with these options. In general, this is what we want since the
//     Kitsune frontends have different defaults for certain options such as
//     -ffp-contract=
//
// Ideally, we would like to share the option definitions with tools such as opt
// and lld, but each tool uses a slightly different tablegen syntax, so
// everything has to be kept in sync manually.
//
//===----------------------------------------------------------------------===//

// The kitsune_Group should be first in this list. This ensures that it appears
// first in the Sphinx-generated HTML documentation.
def kitsune_Group :
    OptionGroup<"<Kitsune-specific option group>">,
    DocName<"Kitsune options">,
    DocBrief<[{Options that are specific to the Kitsune compiler drivers i.e.
               those named kit*. These cannot be used with other drivers such as
               clang or flang.}]>;

// KitsuneOption - This is an option that is only available in one or more of
// the kitsune frontends, kitcc, kit++, and kitfc.
def KitsuneOption : OptionVisibility;

let Group = kitsune_Group in {

def kokkos : Flag<["--"], "kokkos">,
    Visibility<[KitsuneOption, ClangOption, CC1Option]>,
    HelpText<"Enable custom parsing and code generation of some Kokkos "
             "constructs">;

def fkokkos : Flag<["-"], "fkokkos">, Alias<kokkos>,
    Visibility<[KitsuneOption, ClangOption, CC1Option]>,
    HelpText<"DEPRECATED: Use --kokkkos">;

def kokkos_no_init : Flag<["--"], "kokkos-no-init">,
    Visibility<[KitsuneOption, ClangOption, CC1Option]>,
    HelpText<"Skip generation of Kokkos initialize and finalize calls">;

def fkokkos_no_init : Flag<["-"], "fkokkos-no-init">, Alias<kokkos_no_init>,
    Visibility<[KitsuneOption, ClangOption, CC1Option]>,
    HelpText<"DEPRECATED: Use --kokkos-no-init">;

def kitrt_verbose: Flag<["--"], "kitrt-verbose">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    HelpText<"Enable verbose mode in kitsune's runtime">;

def tapir_EQ : Joined<["--"], "tapir=">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    MetaVarName<"<target>">,
    Values<"cuda,custom,hip,nolo,opencilk,pthreads,serial">,
    HelpText<"The primary tapir target">;

def tapir_plugin_EQ : Joined<["--"], "tapir-plugin=">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    MetaVarName<"<dsopath>">, Flags<[NoArgumentUnused]>,
    HelpText<"Load a plugin containing a custom tapir target from a shared "
             "object file">;

def ftapir_EQ : Joined<["-"], "ftapir=">, Alias<tapir_EQ>,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    HelpText<"DEPRECATED: Use --tapir=">;

def tapir_verbose: Flag<["--"], "tapir-verbose">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    HelpText<"Enable verbose mode in all tapir targets. This also enables "
             "verbose mode in the runtime i.e. it implies --kitrt-verbose">;

def tapir_gpu_tpb_EQ: Joined<["--"], "tapir-gpu-tpb=">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    MetaVarName<"<N>">,
    HelpText<"Use a fixed number of threads per block for all GPU kernel "
             "launches unless overridden with pragmas. If this is not provided "
             "the threads per block will be calculated by Kitsune's runtime. "
             "Can be at most 1024">;

def tapir_gpu_max_tpb_EQ: Joined<["--"], "tapir-gpu-max-tpb=">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    MetaVarName<"<N>">,
    HelpText<"The maximum number of threads per block to launch. If this is "
             "not provided, Kitsune's runtime is free to launch as many "
             "threads per block as it sees fit">;

def tapir_gpu_prefetch : Flag<["--"], "tapir-gpu-prefetch">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    HelpText<"Enable generation of prefetch calls to move managed memory "
             "between host and GPU (default)">;

def tapir_gpu_no_prefetch : Flag<["--"], "tapir-gpu-no-prefetch">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    HelpText<"Disable generation of prefetch calls to move managed memory "
             "between host and GPU">;

def tapir_cuda_arch_EQ : Joined<["--"], "tapir-cuda-arch=">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    Default<"">, MetaVarName<"<arch>">,
    HelpText<"The NVIDIA GPU architecture for the cuda tapir target">;

// This is generated by the driver from --tapir-cuda-arch=.
def tapir_cuda_virt_arch_EQ : Joined<["--"], "tapir-cuda-virt-arch=">,
    Visibility<[KitsuneOption, CC1Option, FC1Option]>,
    MetaVarName<"<arch>">,
    HelpText<"The NVIDIA GPU virtual architecture for the current cuda "
             "architecture">;

// This is generated by the driver from --tapir-cuda-arch=. This generally only
// contains the PTX version to be used.
def tapir_cuda_features_EQ : Joined<["--"], "tapir-cuda-features=">,
    Visibility<[KitsuneOption, CC1Option, FC1Option]>,
    MetaVarName<"<features>">,
    HelpText<"The target features for the current cuda architecture. This is "
             "of the form '+feature1,+feature2,...'">;

// This is generated by the driver.
def tapir_cuda_runtime_bc_EQ : Joined<["--"], "tapir-cuda-runtime-bc=">,
    Visibility<[KitsuneOption, CC1Option, FC1Option]>,
    MetaVarName<"<file>">,
    HelpText<"Path to the cuda runtime bitcode file">;

def tapir_hip_arch_EQ : Joined<["--"], "tapir-hip-arch=">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    Default<"">, MetaVarName<"<arch>">,
    HelpText<"The AMDGPU architecture for the hip tapir target">;

// FIXME: The default value of this option is set to 'on', but it really ought
// to be 'any'. More details follow:
//
// This is set to 'on' by default temporarily. Prior to LLVM 20, the only way
// to control xnack was with the -mllvm -enable-sramecc=[[true]|false], which
// has been removed. The default here should be switched to 'any' after a
// transition period.
def tapir_hip_sramecc_EQ : Joined<["--"], "tapir-hip-sramecc=">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    Default<"on">, MetaVarName<"<status>">,
    Values<"on,off,any">,
    HelpText<"Setting this to 'on' enables xnack (equivalent to sramecc+), "
             "'off' disables sramecc (equivalent to sramecc-). Either of these "
             "values will generate code that will only work on a processor "
             "with a matching value of sramecc. The device on which the code "
             "is to be executed must support sramecc if enabled. Setting this "
             "to 'any' will leave the value of sramecc unspecified. This will "
             "generate code that can be executed in a process with either "
             "setting of sramecc enabled">;

// FIXME: The value of this option is set to 'on', but it really ought to be
// 'any'. More details follow:
//
// This is set to 'on' by default temporarily. Prior to LLVM 20, the only way
// to control xnack was with the -mllvm -enable-xnack=[[true]|false], which
// has been removed. The default should be switched to 'any' after a transition
// period.
def tapir_hip_xnack_EQ : Joined<["--"], "tapir-hip-xnack=">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    Default<"on">, MetaVarName<"<status>">,
    Values<"on,off,any">,
    HelpText<"Setting this to 'on' enables xnack (equivalent to xnack+), 'off'"
             "disables xnack (equivalent to xnack-). Either of these values "
             "will generate code that will only work on a processor with a "
             "matching value of xnack. The device on which the code is to be "
             "executed must support xnack if enabled. Setting this to 'any' "
             "will leave the value of xnack unspecified. This will generate "
             "code that can be executed in a process with either setting of "
             "xnack replay enabled">;

// This is generated by the driver from --tapir-hip-arch=.
def tapir_hip_features_EQ : Joined<["--"], "tapir-hip-features=">,
    Visibility<[KitsuneOption, CC1Option, FC1Option]>,
    HelpText<"The target features for the AMDGPU. This is determined from the "
             "architecture of the GPU for which code is being generated. But "
             "it is provided in case it needs to be overridden">;

// This is generated by the driver from the value of --tapir-hip-arch= and any
// other options that have been passed.
def tapir_hip_runtime_bcs_EQ : Joined<["--"], "tapir-hip-runtime-bcs=">,
    Visibility<[KitsuneOption, CC1Option, FC1Option]>,
    MetaVarName<"<files>">,
    HelpText<"A semicolon separated list of bitcode files needed by the hip "
             "tapir target">;

def tapir_hip_rocm_abi_EQ : Joined<["--"], "tapir-hip-abi-version=">,
    Alias<mcode_object_version_EQ>,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    MetaVarName<"<version>">,
    HelpText<"Alias for -mcode-object-version=">;

def tapir_hip_wavefront64 : Flag<["--"], "tapir-hip-wavefront64">,
    Alias<mwavefrontsize64>,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    HelpText<"Alias for -mwavefrontsize64">;

def tapir_hip_wavefront32 : Flag<["--"], "tapir-hip-wavefront32">,
    Alias<mno_wavefrontsize64>,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    HelpText<"Alias for -mno-wavefrontsize64">;

// There is a fair bit of logic that looks for the correct bitcode file to use
// depending on the current target. While that could be internalized, having
// this option is very useful for the tests because we can point to a dummy
// bitcode file and stop it from producing the "wrong" error (for example, to
// ensure that the frontend does not error out on supported platforms).
def tapir_opencilk_runtime_bc_EQ : Joined<["--"], "tapir-opencilk-runtime-bc=">,
    Visibility<[KitsuneOption, CC1Option, FC1Option]>,
    MetaVarName<"<file>">,
    HelpText<"Path to OpenCilk ABI bitcode file">;

def tapir_opencilk_abi_bc_EQ : Joined<["-"], "tapir-opencilk-abi-bc=">,
    Alias<tapir_opencilk_runtime_bc_EQ>,
    Visibility<[KitsuneOption, CC1Option, FC1Option]>,
    HelpText<"DEPRECATED: Use --tapir-opencilk-runtime-bc=">;

def tapir_lld_EQ : Joined<["--"], "tapir-lld=">,
    Visibility<[KitsuneOption, CC1Option, FC1Option]>,
    MetaVarName<"dir">,
    HelpText<"Absolute path to lld that was built with Kitsune. This may be "
             "the path to the linker in the build directory">;

def static_libkitrt : Flag<["-"], "static-libkitrt">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    HelpText<"Statically link Kitsune's runtime">;

// It would be nice if we could use one of the Bool*Option classes, but they
// expect the option to be marshalled directly. We don't have that enabled
// right now, so we have to do it the ugly way.
def fstripmine : Flag<["-"], "fstripmine">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    HelpText<"Enable the loop strip-mining pass (Kitsune only)">;

def fno_stripmine : Flag<["-"], "fno-stripmine">,
    Visibility<[KitsuneOption, ClangOption, CC1Option, FlangOption, FC1Option]>,
    HelpText<"Disable the loop strip-mining pass (Kitsune only)">;

} // Group = kitsune_Group
