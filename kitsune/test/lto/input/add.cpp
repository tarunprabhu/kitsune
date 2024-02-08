#include <kitsune.h>

void vecadd(kitsune::mobile_ptr<double> c, const kitsune::mobile_ptr<double> a,
            const kitsune::mobile_ptr<double> b, size_t n) {
  // clang-format off
  forall(size_t i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
  // clang-format on
}
