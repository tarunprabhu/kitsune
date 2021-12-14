#include <cstdio>
#include <cassert>
#include <kitsune.h>

int main (int argc, char* argv[]) {
  float A[]={1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0};
  float B[]={5.0, 4.75, 4.50, 4.25, 4.0, 3.75, 3.5, 3.25, 3.0};
  float C[]={0.0, 0.00, 0.00, 0.00, 0.0, 0.00, 0.0, 0.00, 0.0};
  size_t INDEX[] = {8, 7, 6, 5, 4, 3, 2, 1, 0};

  forall(auto i : INDEX) {
  //forall(size_t i = 0; i < 9; i++) {
    fprintf(stdout, "i=%lu, INDEX[%lu] = %lu \n", i, i, INDEX[i]);
    assert(i < 9 && "this should never happen!");
    C[i] = A[i] + B[i];
  }

  fprintf(stdout, " %f : %f\n", C[0],C[8]);
  return 0;
}
