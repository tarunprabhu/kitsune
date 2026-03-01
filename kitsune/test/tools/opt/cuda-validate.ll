; REQUIRES: kitsune-cuda
;
; Check that the values of the command line options used by the cuda tapir
; target are validated by opt.
;
; ------------------------------------------------------------------------------
; RUN: not opt --tapir=cuda --tapir-cuda-arch= \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=ARCH
;
; ARCH: error: option '--tapir-cuda-arch' has invalid value ''
;
; ------------------------------------------------------------------------------
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc= \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC
;
; BC: error: option '--tapir-cuda-runtime-bc' has invalid value ''
;
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=noexist \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC-NOEXIST
;
; BC-NOEXIST: error: could not parse LLVM file
; BC-NOEXIST-SAME: Could not open input file
; BC-NOEXIST-SAME: No such file or directory
;
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=%S/input/bogus.ll \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC-CONTENTS
;
; BC-CONTENTS: error: could not parse LLVM file
; BC-CONTENTS-SAME: expected top-level entity
;
; ------------------------------------------------------------------------------
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:         --tapir-cuda-virt-arch= \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=VIRTARCH
;
; VIRTARCH: error: option '--tapir-cuda-virt-arch' has invalid value ''
;
; ------------------------------------------------------------------------------
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:         --tapir-cuda-features= \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=FEATURES
;
; FEATURES: error: option '--tapir-cuda-features' has invalid value ''
;
; ------------------------------------------------------------------------------
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:         --tapir-gpu-tpb= \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=TPB
;
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:         --tapir-gpu-tpb=ten \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=TPB
;
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:         --tapir-gpu-tpb=0 \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=RANGE
;
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:         --tapir-gpu-tpb=1025 \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=RANGE
;
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:         --tapir-gpu-max-tpb= \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=TPB
;
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:         --tapir-gpu-max-tpb=0 \
; RUN:     -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=RANGE
;
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:         --tapir-gpu-max-tpb=1025 \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=RANGE
;
; TPB: for the --tapir-gpu{{(-.+)?}}-tpb option: '{{.*}}' value invalid
; RANGE: error: option '--tapir-gpu{{(-.+)?}}-tpb' has value '{{.+}}' not in range [1,1024]
;
; ------------------------------------------------------------------------------
