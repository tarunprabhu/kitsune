
#include <Kokkos_Core.hpp>
#include <cstdio>

int main (int argc, char* argv[]) {
  //Kokkos::initialize (argc, argv);
  const int n = 10;
  int sum = 11;
  Kokkos::parallel_reduce (n, KOKKOS_LAMBDA (const int i, int& lsum) {
      lsum += i*i;
      printf("%d\n", lsum);
    }, sum);
  printf("final sum = %d\n", sum);
  /*
  printf ("Sum of squares of integers from 0 to %i, "
          "computed in parallel, is %i\n", n - 1, sum);
  // Compare to a sequential loop.
  int seqSum = 0;
  for (int i = 0; i < n; ++i) {
    seqSum += i*i;
  }
  printf ("Sum of squares of integers from 0 to %i, "
          "computed sequentially, is %i\n", n - 1, seqSum);

  Kokkos::finalize ();
  */
  return 0; // (sum == seqSum) ? 0 : -1;
}
