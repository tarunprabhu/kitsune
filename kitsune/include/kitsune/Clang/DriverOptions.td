//===- DriverOptions.td - Options for Kitsune's frontends --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
    DocBrief<[{
Options that are specific to the Kitsune compiler drivers i.e. those named kit*.
These cannot be used with other drivers such as `clang` or `flang`.}]>;

let Group = kitsune_Group in {

//==-----------------------------------------------------------------------===//

// Options visible to both the driver and the underlying compiler for both
// C/C++ and Fortran.
let Visibility = [ClangOption, CC1Option, FlangOption, FC1Option] in {

def kit_instr_EQ : Joined<["--"], "kit-instr=">,
    MetaVarName<"<kinds>">,
    HelpText<"Add Kitsune's instrumentation">,
    DocBrief<[{
Add Kitsune's instrumentation. This is a comma-separated list of 1 or more of
the values specified in the table below.

+------------+-----------------------------+
| generic    | Add generic instrumentation |
+------------+-----------------------------+
| papi       | Add PAPI instrumentation    |
+------------+-----------------------------+
| timer      | Add timers                  |
+------------+-----------------------------+

}]>;

def kit_instr_only_EQ : Joined<["--"], "kit-instr-only=">,
    MetaVarName<"<names>">,
    HelpText<"Comma-separated list of names of entities to instrument">,
    DocBrief<[{
Comma-separated list of names of entities to which to add Kitsune's generic
instrumentation. If this is not given, all entities that match the units given
by `--kit-instr-unit=` will be instrumented.
}]>;

def kit_instr_unit_EQ : Joined<["--"], "kit-instr-unit=">,
    MetaVarName<"<units>">,
    HelpText<"The unit to which Kitsune's instrumentation should be added">,
    DocBrief<[{
Comma-separated list of units to which Kitsune's instrumentation is to be added.
The table below shows the valid values that may appear in this list and
describes where instrumentation will be added.

+------------+-----------------------------------------------------------------+
| loop       | Add instrumentation around tapir loops only                     |
+------------+-----------------------------------------------------------------+
| thread     | Add instrumentation to the body of tapir loops with a parallel, |
|            | CPU-centric tapir target. In this case, the instrumentation's   |
|            | "start" and "stop" functions will be run on each CPU thread     |
+------------+-----------------------------------------------------------------+

In addition, two special values are also accepted. These must be used on their
own - they cannot appear with each other, or in a list. These are:

+------------+-----------------------------------------------------------------+
| all        | Add all supported instrumentation. This adds instrumentation    |
|            | around all supported Kitsune constructs, as well as per-thread  |
|            | instrumentation to the body of tapir loops                      |
+------------+-----------------------------------------------------------------+
| default    | Add instrumentation *around* all supported Kitsune constructs.  |
|            | Per-thread instrumentation is *not* added                       |
+------------+-----------------------------------------------------------------+
}]>;

def kit_instr_papi_EQ : Joined<["--"], "kit-instr-papi=">,
    MetaVarName<"<events>">,
    HelpText<"The PAPI events to record">,
    DocBrief<[{
Comma-separated list of PAPI events to be recorded. This option must be provided
if "papi" is present in the list given to `--kit-instr=`. The table below lists
the convenience names recognized by Kitsune and the PAPI events to which they
correspond.

+-------+----------------+----------------------------------+
| l1d   | `PAPI_L1_DCM`  | Level 1 data cache misses        |
| l2d   | `PAPI_L2_DCM`  | Level 2 data cache misses        |
| l3d   | `PAPI_L3_DCM`  | Level 3 data cache misses        |
| l1i   | `PAPI_L1_ICM`  | Level 1 instruction cache misses |
| l2i   | `PAPI_L2_ICM`  | Level 2 instruction cache misses |
| l3i   | `PAPI_L3_ICM`  | Level 3 instruction cache misses |
| l1t   | `PAPI_L1_TCM`  | Level 1 cache misses             |
| l2t   | `PAPI_L2_TCM`  | Level 2 cache misses             |
| l3t   | `PAPI_L3_TCM`  | Level 3 cache misses             |
| l1ld  | `PAPI_L1_LDM`  | Level 1 load misses              |
| l2ld  | `PAPI_L2_LDM`  | Level 2 load misses              |
| l3ld  | `PAPI_L3_LDM`  | Level 3 load misses              |
| l1st  | `PAPI_L1_STM`  | Level 1 store misses             |
| l2st  | `PAPI_L2_STM`  | Level 2 store misses             |
| l3st  | `PAPI_L3_STM`  | Level 3 store misses             |
| tlbd  | `PAPI_TLB_DM`  | Data TLB misses                  |
| tlbi  | `PAPI_TLB_IM`  | Instruction TLB misses           |
| tlbt  | `PAPI_TLB_TL`  | Total TLB misses                 |
| inst  | `PAPI_TOT_INS` | Instructions completed           |
| ins   | `PAPI_TOT_INS` | Instructions completed           |
| vec   | `PAPI_VEC_INS` | Vector/SIMD instructions         |
| ld    | `PAPI_LD_INS`  | Load instructions completed      |
| st    | `PAPI_SR_INS`  | Store instructions completed     |
| br    | `PAPI_BR_INS`  | Branch instructions completed    |
| int   | `PAPI_INT_INS` | Integer instructions completed   |
| fp    | `PAPI_FP_INS`  | Floating point instructions      |
| fma   | `PAPI_FMA_INS` | FMA instructions completed       |
| stall | `PAPI_RES_STL` | Cycles stalled on any resource   |
| cyc   | `PAPI_TOT_CYC` | Total cycles                     |
| ref   | `PAPI_REF_CYC` | Reference clock cycles           |
+-------+----------------+----------------------------------+

In addition to these, any PAPI event can be specified by dropping the leading
`PAPI_` prefix from the event enum, and converting everything else to lowercase.
For example, "l1_dcm", and "ref_cyc" correspond to the `PAPI_L1_DCM` and
`PAPI_REF_CYC` events respectively.

The same event must not be specified more than once in the list given to this
option, even if different, but equivalent spelling are used. The compiler will
not issue any warnings if this is done. However, this may result in unexpected
behavior at run-time.

.. code::shell

  --kit-instr-papi="ins,tot_ins"

An empty list is not allowed. If "papi" is not in the list given to
`--kit-instr=`, this option has no effect.
}]>;

def tapir_EQ : Joined<["--"], "tapir=">,
    MetaVarName<"<target>">,
    Values<"cuda,custom,hip,nolo,opencilk,pthreads,serial">,
    HelpText<"The primary tapir target">;

def tapir_plugin_EQ : Joined<["--"], "tapir-plugin=">,
    MetaVarName<"<dsopath>">, Flags<[NoArgumentUnused]>,
    HelpText<"Load a plugin containing a custom tapir target from a shared "
             "object file">;

def ftapir_EQ : Joined<["-"], "ftapir=">, Alias<tapir_EQ>,
    HelpText<"DEPRECATED: Use --tapir=">;

def tapir_gpu_prefetch : Flag<["--"], "tapir-gpu-prefetch">,
    HelpText<"Enable generation of prefetch calls to move managed memory "
             "between host and GPU (default)">;

def tapir_gpu_no_prefetch : Flag<["--"], "tapir-gpu-no-prefetch">,
    HelpText<"Disable generation of prefetch calls to move managed memory "
             "between host and GPU">;

def tapir_cuda_arch_EQ : Joined<["--"], "tapir-cuda-arch=">,
    Default<"">, MetaVarName<"<arch>">,
    HelpText<"The NVIDIA GPU architecture for the cuda tapir target">;

def tapir_hip_arch_EQ : Joined<["--"], "tapir-hip-arch=">,
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
    Default<"on">, MetaVarName<"<status>">,
    Values<"on,off,any">,
    HelpText<"sramecc settings to use when generating AMDGPU code">,
    DocBrief<[{
Setting this to 'on' enables xnack (equivalent to sramecc+), 'off' disables
sramecc (equivalent to sramecc-). Either of these values will generate code that
will only work on a processor with a matching value of sramecc. The device on
which the code is to be executed must support sramecc if enabled. Setting this
to 'any' will leave the value of sramecc unspecified. This will generate code
that can be executed in a process with either setting of sramecc enabled}]>;

// FIXME: The value of this option is set to 'on', but it really ought to be
// 'any'. More details follow:
//
// This is set to 'on' by default temporarily. Prior to LLVM 20, the only way
// to control xnack was with the -mllvm -enable-xnack=[[true]|false], which
// has been removed. The default should be switched to 'any' after a transition
// period.
def tapir_hip_xnack_EQ : Joined<["--"], "tapir-hip-xnack=">,
    Default<"on">, MetaVarName<"<status>">,
    Values<"on,off,any">,
    HelpText<"xnack settings to use when generating AMDGPU code">,
    DocBrief<[{
Setting this to 'on' enables xnack (equivalent to xnack+), 'off' disables xnack
(equivalent to xnack-). Either of these values will generate code that will only
work on a processor with a matching value of xnack. The device on which the code
is to be executed must support xnack if enabled. Setting this to 'any' will
leave the value of xnack unspecified. This will generate code that can be
executed in a process with either setting of xnack replay enabled}]>;

def tapir_hip_rocm_abi_EQ : Joined<["--"], "tapir-hip-abi-version=">,
    Alias<mcode_object_version_EQ>,
    MetaVarName<"<version>">,
    HelpText<"Alias for -mcode-object-version=">;

def tapir_hip_wavefront64 : Flag<["--"], "tapir-hip-wavefront64">,
    Alias<mwavefrontsize64>,
    HelpText<"Alias for -mwavefrontsize64">;

def tapir_hip_wavefront32 : Flag<["--"], "tapir-hip-wavefront32">,
    Alias<mno_wavefrontsize64>,
    HelpText<"Alias for -mno-wavefrontsize64">;

def static_libkitrt : Flag<["-"], "static-libkitrt">,
    HelpText<"Statically link Kitsune's runtime">;

// It would be nice if we could use one of the Bool*Option classes, but they
// expect the option to be marshalled directly. We don't have that enabled
// right now, so we have to do it the ugly way.
def fstripmine : Flag<["-"], "fstripmine">,
    HelpText<"Enable the loop strip-mining pass (Kitsune only)">;

def fno_stripmine : Flag<["-"], "fno-stripmine">,
    HelpText<"Disable the loop strip-mining pass (Kitsune only)">;

} // Visibility = [ClangOption, CC1Option, FlangOption, FC1Option]

//===----------------------------------------------------------------------===//

// C/C++-only options that are visible to both the driver and `-cc1`.
let Visibility = [ClangOption, CC1Option] in {

def kokkos : Flag<["--"], "kokkos">,
    HelpText<"Enable custom parsing and code generation of some Kokkos "
             "constructs">;

def fkokkos : Flag<["-"], "fkokkos">, Alias<kokkos>,
    HelpText<"DEPRECATED: Use --kokkkos">;

def kokkos_no_init : Flag<["--"], "kokkos-no-init">,
    HelpText<"Skip generation of Kokkos initialize and finalize calls">;

def fkokkos_no_init : Flag<["-"], "fkokkos-no-init">, Alias<kokkos_no_init>,
    HelpText<"DEPRECATED: Use --kokkos-no-init">;

} // Visibility = [ClangOption, CC1Option]

//===----------------------------------------------------------------------===//

// Options that are only visible to the underlying compiler. These are nearly
// always generated by the driver, though they can be specified using
// `-Xclang`/`-Xflang` if the user really knows what they are doing.
let Visibility = [CC1Option, FC1Option] in {

// This is generated by the driver from --tapir-cuda-arch=.
def tapir_cuda_virt_arch_EQ : Joined<["--"], "tapir-cuda-virt-arch=">,
    MetaVarName<"<arch>">,
    HelpText<"The NVIDIA GPU virtual architecture for the current cuda "
             "architecture">;

// This is generated by the driver from --tapir-cuda-arch=. This generally only
// contains the PTX version to be used.
def tapir_cuda_features_EQ : Joined<["--"], "tapir-cuda-features=">,
    MetaVarName<"<features>">,
    HelpText<"The target features for the current cuda architecture. This is "
             "of the form '+feature1,+feature2,...'">;

// This is generated by the driver.
def tapir_cuda_runtime_bc_EQ : Joined<["--"], "tapir-cuda-runtime-bc=">,
    MetaVarName<"<file>">,
    HelpText<"Path to the cuda runtime bitcode file">;

// This is generated by the driver from --tapir-hip-arch=.
def tapir_hip_features_EQ : Joined<["--"], "tapir-hip-features=">,
    HelpText<"The target features to use when generating AMDGPU code">,
    DocBrief<[{
This is determined from the architecture of the GPU for which code is being
generated. But it is provided in case it needs to be overridden}]>;

// This is generated by the driver from the value of --tapir-hip-arch= and any
// other options that have been passed.
def tapir_hip_runtime_bcs_EQ : Joined<["--"], "tapir-hip-runtime-bcs=">,
    MetaVarName<"<files>">,
    HelpText<"A semicolon separated list of bitcode files needed by the hip "
             "tapir target">;

// There is a fair bit of logic that looks for the correct bitcode file to use
// depending on the current target. While that could be internalized, having
// this option is very useful for the tests because we can point to a dummy
// bitcode file and stop it from producing the "wrong" error (for example, to
// ensure that the frontend does not error out on supported platforms).
def tapir_opencilk_runtime_bc_EQ : Joined<["--"], "tapir-opencilk-runtime-bc=">,
    MetaVarName<"<file>">,
    HelpText<"Path to OpenCilk ABI bitcode file">;

def tapir_opencilk_abi_bc_EQ : Joined<["-"], "tapir-opencilk-abi-bc=">,
    Alias<tapir_opencilk_runtime_bc_EQ>,
    HelpText<"DEPRECATED: Use --tapir-opencilk-runtime-bc=">;

def tapir_lld_EQ : Joined<["--"], "tapir-lld=">,
    MetaVarName<"dir">,
    HelpText<"Absolute path to lld that was built with Kitsune. This may be "
             "the path to the linker in the build directory">;

} // Visibility = [CC1Option, FC1Option]

// -----------------------------------------------------------------------------

} // Group = kitsune_Group
