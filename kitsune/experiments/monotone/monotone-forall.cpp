// See the README file for details.
#include "kitrt/kitcuda/cuda.h"
#include "kitsune/timer.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <kitsune.h>

using namespace std;
using namespace kitsune;

const size_t DEFAULT_ARRAY_SIZE = 1024 * 1024 * 128;
const unsigned int DEFAULT_ITERATIONS = 10;

template <typename T> T *alloc(int N) {
  return (T *)__kitrt_cuMemAllocManaged(sizeof(T) * N);
}

template <typename T> void dealloc(T *array) {
  __kitrt_cuMemFree((void *)array);
}

template <typename T>
void zeros(T* arr, size_t N) {
  for(size_t i = 0; i < N; i++)
    arr[i] = 0;

  // we've updated the data array -- flag it for
  // prefetching the next time we launch a kernel
  // (forall loop) that uses it...
  __kitrt_cuMemNeedsPrefetch((void *)arr);
}

template <typename T> size_t check(const T *arr, size_t N, unsigned iterations) {
  for (size_t i = 0; i < N; ++i) {
    if (arr[i] != iterations)
      return i;
  }
  // In the process of the check we've paged 'data1'
  // back to the host.  Flag it as needing a prefetch
  // again; this forces all pages to be moved prior
  // to a kernel launch that is dependent upon it.
  // This highlights an issue with UVM usage as we
  // would really like a copy resident on the GPU and
  // here to be checked on the CPU...
  __kitrt_cuMemNeedsPrefetch((void *)arr);
  return -1;
}

template <typename T> void parallel_incr(T *arr, int N, unsigned iterations) {
  forall(int i = 0; i < N; i++) arr[i]++;
}

int main(int argc, char **argv) {
  size_t array_size = DEFAULT_ARRAY_SIZE;
  unsigned int iterations = DEFAULT_ITERATIONS;
  if (argc >= 2)
    array_size = atol(argv[1]);

  if (argc == 3)
    iterations = atoi(argv[2]);

  fprintf(stdout, "array size: %ld\n", array_size);
  fprintf(stdout, "iterations = %d\n", iterations);

  int *arr = alloc<int>(array_size);
  zeros(arr, array_size);

  auto start = chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    auto incr_start = chrono::steady_clock::now();
    parallel_incr(arr, array_size, iterations);
    auto incr_end = chrono::steady_clock::now();
    cout << "Incr time: "
         << chrono::duration<double>(incr_end - incr_start).count() << endl;
  }

  // This will page fault GPU-side pages back to the
  // CPU -- page faults should impact our overall
  // performance.
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
  return 0;
}
