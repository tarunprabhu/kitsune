
#include <Kokkos_Core.hpp>
#include <cstdio>

//
// First reduction (parallel_reduce) example:
//   1. Start up Kokkos
//   2. Execute a parallel_reduce loop in the default execution space,
//      using a C++11 lambda to define the loop body
//   3. Shut down Kokkos
//
// This example only builds if C++11 is enabled.  Compare this example
// to 02_simple_reduce, which uses a functor to define the loop body
// of the parallel_reduce.
//

int main (int argc, char* argv[]) {
  //Kokkos::initialize (argc, argv);
  const int n = 10;
  int sum = 0;
  Kokkos::parallel_reduce (n, KOKKOS_LAMBDA (const int i, int& lsum) {
      lsum += i*i;
      printf("%d\n", lsum);
    }, sum);
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
