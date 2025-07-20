; Check that the kit-embs pass creates global variables for the embedded bitcode
; and fat binary. All functions in the host that contain the kit_device
; attribute must have been cloned into the embedded module. Any global values
; transitively used by the kernel functions must also be present in the embedded
; device module.
;
; RUN: mkdir -p %t
; RUN: opt --tapir=cuda -passes='kit-embs,verify' -S -o %t/with-embs.ll %s
; RUN: cat %t/with-embs.ll \
; RUN:     | FileCheck %s -check-prefix HOST
; RUN: cat %t/with-embs.ll \
; RUN:     | kit-mbc -S -o - \
; RUN:     | FileCheck %s -check-prefix DEVICE
;
; HOST: @{{.+}} = {{.*}}constant [{{[0-9]+}} x i8] c"BC{{.+}}"
; HOST-SAME: #[[BC:[0-9]+]]
; HOST: @{{.+}} = {{.*}}constant [0 x i8] zeroinitializer
; HOST-SAME: #[[FB:[0-9]+]]
; HOST-DAG: #[[BC]] = { kit_bc kit_tt(2) }
; HOST-DAG: #[[FB]] = { kit_fb kit_tt(2) }
;
; DEVICE-NOT: @gh
; DEVICE-NOT: @h4
; DEVICE: @gd = dso_local global i32 0
; DEVICE-DAG: define void @d0
; DEVICE-DAG: define void @d1
; DEVICE-DAG: define void @d2
; DEVICE-DAG: define void @d3

@gd = global i32 120
@gh = global i32 240

define void @d0() #0 {
  ret void
}

define void @d1() #1 {
  call void @d2()
  ret void
}

define void @d2() {
  call void @d3(ptr @gd)
  ret void
}

define void @d3(ptr %0) {
  ret void
}

define void @h4() {
  ret void
}

attributes #0 = { kit_device }
attributes #1 = { kit_device }
