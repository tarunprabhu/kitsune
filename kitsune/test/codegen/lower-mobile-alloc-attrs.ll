; Check that the attributes are preserved when lowering llvm.kit.mobile.alloc
; intrinsic. This test is a special case because the intrinsic is handled as a
; special case in the lowering code.
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=none -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes NONE,ATTRS %s
;
; NONE: define {{.+}} @allocate(i64 %[[N:.+]])
; NONE-NEXT: call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %[[N]]){{$}}
; NONE-NEXT: call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %[[N]]) #[[ATTR:[0-9]+]]
; NONE-NEXT: call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 noundef %[[N]]){{$}}
; NONE-NEXT: call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 noundef %[[N]]) #[[ATTR]]
; NONE-NEXT: ret ptr addrspace(67) null
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes SERIAL,ATTRS %s
;
; SERIAL: define {{.+}} @allocate(i64 %[[N:.+]])
; SERIAL: call noalias ptr @malloc(i64 %[[N]]){{$}}
; SERIAL: call noalias ptr @malloc(i64 %[[N]]) #[[ATTR:[0-9]+]]
; SERIAL: call noalias ptr @malloc(i64 noundef %[[N]]){{$}}
; SERIAL: call noalias ptr @malloc(i64 noundef %[[N]]) #[[ATTR]]
; SERIAL: ret ptr addrspace(67) null
;
; ------------------------------------------------------------------------------
;
; ATTRS: attributes #[[ATTR]] = { "custom-attribute" }
;
; ------------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

define noalias ptr addrspace(67) @allocate(i64 %n) {
  %1 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %n)
  %2 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %n) #1
  %3 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 noundef %n)
  %4 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 noundef %n) #1

  ret ptr addrspace(67) null
}

attributes #1 = { "custom-attribute" }
