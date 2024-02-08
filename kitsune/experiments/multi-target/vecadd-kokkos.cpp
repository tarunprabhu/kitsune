#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <kitsune.h>

using namespace kitsune;
using namespace std;

const size_t ARRAY_SIZE = 1024 * 1024 * 256;

void cpu_fill(mobile_ptr<float> data_p, size_t N, int inc) {
  // FIXME: Kokkos cannot deal with the mobile_ptr type for ... reasons.
  // We could try to find a way to make that type Kokkos-friendly, or just wait
  // until the [mobile_ptr] attribute is implemented which should make
  // Kokkos happy.
  float *data = data_p.get();

  // clang-format off
  Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
    data[i] = float(i + inc);
  });
  // clang-format on
}

void gpu_fill(mobile_ptr<float> data_p, size_t N, int inc) {
  // FIXME: Kokkos cannot deal with the mobile_ptr type for ... reasons.
  // We could try to find a way to make that type Kokkos-friendly, or just wait
  // until the [mobile_ptr] attribute is implemented which should make
  // Kokkos happy.
  float *data = data_p.get();

  // clang-format off
  [[tapir::target("cuda")]]
  Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
    data[i] = float(i + inc);
  });
  // clang-format on
}

int main(int argc, char *argv[]) {
  using namespace std;
  size_t size = ARRAY_SIZE;

  unsigned int iterations = 10;
  if (argc >= 2)
    size = atol(argv[1]);
  if (argc == 3)
    iterations = atoi(argv[2]);

  cout << setprecision(5);
  cout << "\n";
  cout << "---- multi-target vector addition (kokkos) ----\n"
       << "  Vector size: " << size << " elements.\n\n";
  cout << "  Allocating arrays..." << std::flush;

  size_t error_count = 0;
  bool found_error = false;

  Kokkos::initialize(argc, argv);
  {
    mobile_ptr<float> A(size);
    mobile_ptr<float> B(size);
    mobile_ptr<float> C(size);
    cout << "  done.\n\n";

    for (int t = 0; t < iterations; t++) {
      // clang-format off
      spawn fill_a {
        cpu_fill(A, size, t);
      }
      spawn fill_b {
        gpu_fill(B, size, t);
      }
      sync fill_a;
      sync fill_b;

      // FIXME: Kokkos cannot deal with the mobile_ptr type for ... reasons.
      // We could try to find a way to make that type Kokkos-friendly, or just
      // wait until the [mobile_ptr] attribute is implemented which should
      // make Kokkos happy.
      float* a = A.get();
      float* b = B.get();
      float* c = C.get();

      spawn add_gpu {
        [[tapir::target("cuda")]]
        Kokkos::parallel_for(size / 2, KOKKOS_LAMBDA(const int i) {
          c[i] = a[i] + b[i];
        });
      }

      spawn add_cpu {
        Kokkos::parallel_for(size / 2, KOKKOS_LAMBDA(const int i) {
          c[i + size / 2] = a[i + size / 2] + b[i + size / 2];
        });
      }
      sync add_gpu;
      sync add_cpu;
      // clang-format on

      cout << "  checking result..." << std::flush;
      error_count = 0;
      for (size_t i = 0; i < size; i++) {
        float sum = A[i] + B[i];
        if (C[i] != sum)
          error_count++;
      }
      if (error_count > 0) {
        cout << "  incorrect result found!\n";
        found_error = true;
      } else
        cout << "  ok\n";
    }
  }
  Kokkos::finalize();

  return int(found_error);
}
