; Check that attributes are preserved when lowering the llvm.kit.mobile.free
; intrinsic. This test is a special case because the intrinsic is handled as a
; special case in the lowering code.
;
; ------------------------------------------------------------------------------
; RUN: opt -tapir=none -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes NONE,ATTRS %s
;
; NONE: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; NONE-NEXT: call void @llvm.kit.mobile.free(ptr addrspace(67) %[[P]]){{$}}
; NONE-NEXT: call void @llvm.kit.mobile.free(ptr addrspace(67) %[[P]]) #[[ATTR:[0-9]+]]
; NONE-NEXT: call void @llvm.kit.mobile.free(ptr addrspace(67) nonnull %[[P]]){{$}}
; NONE-NEXT: call void @llvm.kit.mobile.free(ptr addrspace(67) nonnull %[[P]]) #[[ATTR]]
; NONE-NEXT: ret void
;
; ------------------------------------------------------------------------------
; RUN: opt -tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes SERIAL,ATTRS %s
;
; SERIAL: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; SERIAL: call void @free(ptr %{{[^)]+}}){{$}}
; SERIAL: call void @free(ptr %{{[^)]+}}) #[[ATTR:[0-9]+]]
; SERIAL: call void @free(ptr nonnull %{{[^)]+}}){{$}}
; SERIAL: call void @free(ptr nonnull %{{[^)]+}}) #[[ATTR]]
; SERIAL: ret void
;
; ------------------------------------------------------------------------------
;
; ATTRS: attributes #[[ATTR]] = { "custom-attr" }
;
; ------------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

define void @deallocate(ptr addrspace(67) %p) {
  call void @llvm.kit.mobile.free(ptr addrspace(67) %p)
  call void @llvm.kit.mobile.free(ptr addrspace(67) %p) #0
  call void @llvm.kit.mobile.free(ptr addrspace(67) nonnull %p)
  call void @llvm.kit.mobile.free(ptr addrspace(67) nonnull %p) #0
  ret void
}

attributes #0 = { "custom-attr" }
