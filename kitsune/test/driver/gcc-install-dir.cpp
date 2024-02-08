// If kitsune was configured with -DKITSUNE_GCC_INSTALL_DIR, check that the
// path provided there is overridden by configuration files and the value of
// --gcc-install-dir= passed on the command line.
//
// REQUIRES: kitsune-gcc-install-dir
//
// %kitsune_gcc_install_dir is the path to GCC install dir that was provided on
// at configure time. Check that it is used by default.
//
// RUN: %kitcc -### %s 2>&1 \
// RUN:     | grep -E "\"%kitsune_gcc_install_dir/[^\"]+/include\""
//
// RUN: %kitcc -### %s 2>&1 \
// RUN:     | grep -E "\"%kitsune_gcc_install_dir/crtendS.o\""
//
// ----------------------------------------------------------------------------
//
// Check that the GCC install directory that is used can be overridden by a
// command line option
//
// RUN: %kitcc -### %s --gcc-install-dir=%S/input/fake-gcc 2>&1 \
// RUN:     | grep -vE "\"%kitsune_gcc_install_dir/[^\"]+/include\""
//
// RUN: %kitcc -### %s --gcc-install-dir=%S/input/fake-gcc 2>&1 \
// RUN:     | grep -vE "\"%kitsune_gcc_install_dir/crtendS.o\""
//
// RUN: %kitcc -### %s --gcc-install-dir=%S/input/fake-gcc 2>&1 \
// RUN:     | FileCheck %s -check-prefix OVERRIDE
//
// ----------------------------------------------------------------------------
//
// Check that the GCC install directory that is used can be overridden by a
// configuration file
//
// RUN: mkdir -p %t
// RUN: echo "--gcc-install-dir=%S/input/fake-gcc" > %t/gcc_install_dir.cfg
// RUN: %kitcc -### %s --config=%t/gcc_install_dir.cfg 2>&1 \
// RUN:     | FileCheck %s -check-prefix OVERRIDE
//
// RUN: %kitcc -### %s --config=%t/gcc_install_dir.cfg 2>&1 \
// RUN:     | grep -vE "\"%kitsune_gcc_install_dir/[^\"]+/include\""
//
// RUN: %kitcc -### %s --config=%t/gcc_install_dir.cfg 2>&1 \
// RUN:     | grep -vE "\"%kitsune_gcc_install_dir/crtendS.o\""
//
// OVERRIDE: -cc1
// OVERRIDE-SAME: "{{.+}}/input/fake-gcc/{{[^\"]*}}/include"
// OVERRIDE: -dynamic-linker
// OVERRIDE-SAME: "-L{{.+}}/fake-gcc"
// OVERRIDE-SAME: /input/fake-gcc/crtendS.o
