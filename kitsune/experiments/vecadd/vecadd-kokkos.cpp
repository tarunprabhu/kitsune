#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>

typedef Kokkos::DualView<float*, Kokkos::LayoutRight,
                         Kokkos::DefaultExecutionSpace>
    DualViewVector;

void random_fill(DualViewVector &data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data.h_view(i) = rand() / (float)RAND_MAX;
}

int main (int argc, char* argv[]) {
  using namespace std;
  size_t size = 1024 * 1024 * 256;
  unsigned int iterations = 10;
  if (argc >= 2)
    size = atol(argv[1]);
  if (argc == 3)
    iterations = atoi(argv[2]);  
  
  cout << setprecision(5);
  cout << "\n";
  cout << "---- vector addition benchmark (kokkos) ----\n"
       << "  Vector size: " << size << " elements.\n\n";
  cout << "  Allocating arrays and filling with random values..." 
       << std::flush;

  Kokkos::initialize(argc, argv); {
    DualViewVector A = DualViewVector("A", size);
    DualViewVector B = DualViewVector("B", size);
    DualViewVector C = DualViewVector("C", size);
    
    random_fill(A, size);
    A.modify_host();

    random_fill(B, size);
    B.modify_host();
    cout << "  done.\n\n";

    double elapsed_time;
    double min_time = 100000.0;
    double max_time = 0.0;    
    for(int t = 0; t < iterations; t++) {
      auto start_time = chrono::steady_clock::now();
      A.sync_device();
      B.sync_device();
      C.sync_device();
      Kokkos::parallel_for(size, KOKKOS_LAMBDA(const int i) {
        C.d_view(i) = A.d_view(i) + B.d_view(i);
      });
      C.modify_device();
      Kokkos::fence();
      auto end_time = chrono::steady_clock::now();
      elapsed_time = chrono::duration<double>(end_time-start_time).count();
      if (elapsed_time < min_time)
	min_time = elapsed_time;
      if (elapsed_time > max_time)
	max_time = elapsed_time;
      cout << "\t" << t << ". iteration time: " << elapsed_time << ".\n";      
    }
    C.sync_host();
    A.sync_host();
    B.sync_host();

    cout << "  Checking final result..." << std::flush;
    size_t error_count = 0;
    for (size_t i = 0; i < size; i++) {
      float sum = A.h_view(i) + B.h_view(i);
      if (C.h_view(i) != sum)
        error_count++;
    }

    if (error_count) {
      cout << "  incorrect result found! (" 
           << error_count << " errors found)\n\n";
      return 1;
    } else {
      cout << "  pass (answers match).\n\n"
           << "  Total time: " << elapsed_time
           << " seconds. (" << size / elapsed_time << " elements/sec.)\n"
           << "*** " << min_time << ", " << max_time << "\n"      	
           << "----\n\n";
    }
  } Kokkos::finalize();
  return 0;
}

