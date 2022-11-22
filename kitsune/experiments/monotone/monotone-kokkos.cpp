#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

using namespace std;

#include "Kokkos_DualView.hpp"

template <typename T>
using View = Kokkos::DualView<T*>;


const size_t DEFAULT_ARRAY_SIZE = 1024 * 1024 * 128;
const unsigned int DEFAULT_ITERATIONS = 10;

#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


#ifdef restrict
#define __restrict restrict
#else
#define __restrict
#endif

template <typename T>
void parallel_incr(View<T> &arr, int N) {
  arr.sync_device();
  Kokkos::parallel_for("incr", N, KOKKOS_LAMBDA(const int &i) {
    arr.d_view(i)++;
  });
  Kokkos::fence();
  arr.modify_device();
}

template <typename T>
void zeros(View<T> &arr, size_t N) {
  for(size_t i = 0; i < N; i++)
    arr.h_view(i) = 0;
  arr.modify_host();
}

template <typename T>
size_t check(View<T> &arr, size_t N, unsigned iterations) {
  arr.sync_host();
  for(size_t i = 0; i < N; ++i) {
    if (arr.h_view(i) != iterations)
      return i;
  }
  return -1;
}


int main(int argc, char** argv)
{
  size_t array_size = DEFAULT_ARRAY_SIZE;
  unsigned int iterations = DEFAULT_ITERATIONS;
  if (argc >= 2)
    array_size = atol(argv[1]);

  if (argc == 3)
    iterations = atoi(argv[2]);

  fprintf(stderr, "array size: %ld\n", array_size);
  fprintf(stderr, "iterations = %d\n", iterations);

  auto start = chrono::steady_clock::now();
  Kokkos::initialize(argc, argv); {
    View<int> arr("arr", array_size);

    zeros(data0, array_size);
    for(int i = 0; i < iterations; i++) {

      auto incr_start = chrono::steady_clock::now();
      parallel_incr(arr, array_size, iterations);
      auto incr_end = chrono::steady_clock::now();
      cout  << "Incr time: "
	    << chrono::duration<double>(incr_end-incr_start).count()
	    << endl;
    }

    size_t badIdx = check(arr, array_size, iterations);
    if (badIdx != -1) {
      fprintf(stderr,
            "Array not incremented correctly at %ld. Expected %d. Got %d.!\n",
            badIdx, iterations, arr[badIdx]);
      return 1;
    }

    auto end = chrono::steady_clock::now();
    cout << "Total time: " << chrono::duration<double>(end - start).count()
         << endl;

  }  Kokkos::finalize();

  return 0;
}
