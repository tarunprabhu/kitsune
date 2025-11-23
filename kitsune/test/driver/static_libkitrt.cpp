// Check that the -static-libkitrt option is handled correctly. The pthreads
// tapir target is guaranteed to be built and always links libkitrt.
//
// ----------------------------------------------------------------------------
//
// RUN: %kitxx -### --tapir=pthreads -O2 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix DYNAMIC
//
// DYNAMIC: -dynamic-linker
// DYNAMIC-SAME: -lkitrt
// DYNAMIC-NOT: -lkitrt_static
//
// ----------------------------------------------------------------------------
// When passing -static-libkitrt, only libkitrt will be linked statically.
// Everything else will be dynamic.
//
// RUN: %kitxx -### --tapir=pthreads -O2 -static-libkitrt %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=STATIC-LIBKITRT
//
// STATIC-LIBKITRT: "-dynamic-linker"
// STATIC-LIBKITRT-SAME: "-Bstatic"
// STATIC-LIBKITRT-SAME: "-lkitrt_static"
// STATIC-LIBKITRT-SAME: "-Bdynamic"
//
// ----------------------------------------------------------------------------
// When using -static, all libraries will be linked statically.
//
// RUN: %kitxx -### --tapir=pthreads -O2 -static %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=STATIC \
// RUN: %if system-linux %{ \
// RUN:     -check-prefix=STATIC-LD \
// RUN: %} \
// RUN: %if system-freebsd || system-darwin %{ \
// RUN:     -check-prefix=STATIC-LLD \
// RUN: %}
//
// STATIC: "-cc1"
// STATIC-LLD-NEXT: "-Bstatic"
// STATIC-LD-NEXT: "-static"
// STATIC-SAME: "-lkitrt_static"
// STATIC-NOT: -Bdynamic
