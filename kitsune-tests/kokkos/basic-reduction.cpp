
#include <Kokkos_Core.hpp>
#include <cstdio>

int main (int argc, char* argv[]) {
  //Kokkos::initialize (argc, argv);
  const int n = 10;
  int sum = 0;
  Kokkos::parallel_reduce (n, KOKKOS_LAMBDA (const int i, int& lsum) {
      lsum += i*i;
      printf("%d\n", lsum);
    }, sum);
  printf("final sum = %d\n", sum);
  //Kokkos::finalize ();
  return 0; // (sum == seqSum) ? 0 : -1;
}
