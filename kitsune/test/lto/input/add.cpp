#include <kitsune.h>

extern "C" void vecadd(double *c, const double *a, const double *b, size_t n) {
  // clang-format off
  forall(size_t i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
  // clang-format on
}
