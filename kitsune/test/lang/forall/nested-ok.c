// Check that some nested parallel forall loops are ok. Currently, we require
// perfect nesting, but we may relax the constraints gradually,
//
// RUN: %kitcc --tapir=nolo -O1 -Xclang -verify -fsyntax-only %s
// expected-no-diagnostics

#include <kitsune.h>

void ext2(int, int);
void ext4(int, int, int, int);

void f0(int n) {
  // clang-format off
  forall (int i = 0; i < n; ++i) {}
  forall (int j = 0; j < n; ++j) {}
  // clang-format on
}

void f01(int n) {
  // clang-format off
  for (int i = 0; i < n; ++i)
    forall (int j = 0; j < n; ++j) {}
  // clang-format on
}

void f1(int n) {
  // clang-format off
  forall (int i = 0; i < n; ++i)
    forall (int j = 0; j < n; ++j)
      ext2(i, j);
  // clang-format on
}

void f2(int n) {
  // clang-format off
  forall (int i = 0; i < n; ++i) {
    forall (int j = 0; j < n; ++j) {
      ext2(i, j);
    }
  }
  // clang-format on
}

// Although we generally only support 3-deep nests for parallel for loops in the
// middle-end, the frontend will accept deeper nests.
void f3(int n) {
  // clang-format off
  forall (int i = 0; i < n; ++i)
    forall (int j = 0; j < n; ++j)
      forall (int k = 0; k < n; ++k)
        forall (int l = 0; l < n; ++l)
          ext4(i, j, k, l);
  // clang-format on
}
