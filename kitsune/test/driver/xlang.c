// Check the -x options that are supported. This list may need to be updated
//
// RUN: not %kitcc -x cuda --tapir=serial -O1 -nocudainc -nocudalib %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix CUDA
// RUN: not %kitcc -x hip --tapir=serial -O1 -nogpuinc -nogpulib %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix HIP
// RUN: not %kitcc -x objective-c --tapir=serial -O1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix OBJC
// RUN: not %kitcc -x cl --tapir=serial -O1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix OPENCL
//
// CUDA: kitsune does not support the Cuda language
// HIP: kitsune does not support the Hip language
// OBJC: kitsune does not support Objective-C
// OPENCL: kitsune does not support OpenCL
