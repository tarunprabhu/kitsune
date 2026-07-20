// Multi-file vector addition benchmark. This is purely to test LTO.

#include <kitsune.h>

extern "C" double random_value(void);
extern "C" void print(const double *c, size_t n);
extern "C" void vecadd(double *c, const double *a, const double *b, size_t n);

static double *alloc(size_t n) {
  double *buf = (double *)malloc(sizeof(double) * n);
  for (size_t i = 0; i < n; ++i)
    buf[i] = random_value();
  return buf;
}

int main(int argc, char *argv[]) {
  size_t n = atol(argv[1]);
  double *a = alloc(n);
  double *b = alloc(n);
  double *c = alloc(n);

  vecadd(c, a, b, n);
  print(c, argc);

  free(c);
  free(b);
  free(a);

  return 0;
}
