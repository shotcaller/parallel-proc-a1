#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main(int argc, char *argv[]) {
  // Check if the number of command-line arguments is correct
  if (argc != 3) {
    // If not, print usage information and exit
    printf("Usage: %s <N> <mode>\n", argv[0]);
    return 1;
  }

  // Parse the command-line arguments (atoi converts string to integer)
  int N = atoi(argv[1]);
  int mode = atoi(argv[2]);

  printf("Running with N: %d, mode: %d\n", N, mode);

  // Allocate contigous memory for matrices A, B, and C (malloc for dynamic allocation)
  double *A = (double *)malloc(N * N * sizeof(double));
  double *B = (double *)malloc(N * N * sizeof(double));
  double *C = (double *)malloc(N * N * sizeof(double));

  // malloc returns pointer to allocated memory or NULL if allocation fails
  if (A == NULL || B == NULL || C == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  // Initialize matrices A and B
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      // (i * N + j) calculates the index for the 1D array representation of the 2D matrix
      A[i * N + j] = sin(i) * cos(j) + sqrt(i + j + 1.0);
      B[i * N + j] = cos(i) * sin(j) + sqrt(i + j + 2.0);

      // Initialize C to zero
      C[i * N + j] = 0.0;
    }
  }

  // Variables for timing and analysis
  double sumC = 0.0; 
  double maxC = 0.0; 
  long long checksum = 0;

  double start_time, kernel_time, total_time;

  start_time = omp_get_wtime(); // Start timing

  if (mode == 0) {
    // Mode 0: Standard matrix multiplication, triple loop for 1D array
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        double temp = 0.0;
        for (int k = 0; k < N; k++) {
          // A[i][k] * B[k][j] mapped to 1D array indices
          temp += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = temp;
      }
    }

    // Record time taken for kernel execution
    kernel_time = omp_get_wtime() - start_time;

    // Calculate sum, max, and checksum for C
    maxC = C[0]; // Initialize maxC to the first element of C

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        double value = C[i * N + j];
        sumC += value; // Accumulate sum of elements in C
        if (value > maxC) {
          maxC = value; // Update maxC if current value is greater
        }
        // checksum as per the assignment formula
        checksum += (long long)(value * 1000.0) % 100000; 
      }
    }

    // Record total time taken for the entire operation
    total_time = omp_get_wtime() - start_time;

    // Print results
    printf("Mode: 0 (Serial Baseline)\n");
    printf("Threads: 1\n");
    printf("Kernel Time: %f s\n", kernel_time);
    printf("Total Time: %f s\n", total_time);
    printf("sumC: %f\n", sumC);
    printf("maxC: %f\n", maxC);
    printf("checksum: %lld\n", checksum);
  }
  else if (mode == 1) {
    // Mode 1: OpenMP threads (Work sharing loops + correct scoping)
    // Change schedule to (dynamic, 16) to test dynamic scheduling with a chunk size of 16
    int i, j, k; // Declare loop variables for private scope
    #pragma omp parallel for default(none) shared(A, B, C, N) private(i, j, k) schedule(dynamic, 16)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        double temp = 0.0;
        for (k = 0; k < N; k++) {
          temp += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = temp;
      }
    }
    kernel_time = omp_get_wtime() - start_time;

    // Analytics with reductions for sum and max
    // sumC and maxC are shared, val is private to each thread
    #pragma omp parallel for default(none) shared(C, N) private(i, j) reduction(+:sumC) reduction(max:maxC)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        double val = C[i * N + j];
        sumC += val; // Accumulate sum of elements in C
        if (val > maxC) maxC = val; // Update maxC if current value is greater
      }
    }

    // Checksum using atomic as a baseline for Mode 1
    #pragma omp parallel for default(none) shared(C, N, checksum) private(i, j) 
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        long long local_val = (long long)(C[i * N + j] * 1000.0) % 100000;
        #pragma omp atomic
        checksum += local_val; // Update checksum atomically
      }
    }
    total_time = omp_get_wtime() - start_time;  

    printf("Mode: 1 (OpenMP Threads)\n");
    printf("Threads: %d\n", omp_get_max_threads());
    printf("Kernel Time: %f s\n", kernel_time);
    printf("Total Time: %f s\n", total_time);
    printf("sumC: %f\n", sumC);
    printf("maxC: %f\n", maxC);
    printf("checksum: %lld\n", checksum);
  }
  else if (mode == 2) {
    // Mode 2: OpenMP threads with collapse(2) for nested loops
    int i, j, k; // Declare loop variables for private scope
    #pragma omp parallel for default(none) shared(A, B, C, N) private(i, j, k) collapse(2)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        double temp = 0.0;
        for (k = 0; k < N; k++) {
          temp += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = temp;
      }
    }
    kernel_time = omp_get_wtime() - start_time;

    // Analytics with reductions for sum and max
    #pragma omp parallel for default(none) shared(C, N) private(i, j) reduction(+:sumC) reduction(max:maxC)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        double val = C[i * N + j];
        sumC += val;
        if (val > maxC) maxC = val;
      }
    }

    // Checksum using atomic for consistency for Mode 2
    #pragma omp parallel for default(none) shared(C, N, checksum) private(i, j)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        long long local_val = (long long)(C[i * N + j] * 1000.0) % 100000;
        #pragma omp atomic
        checksum += local_val;
      }
    }
    total_time = omp_get_wtime() - start_time;

    printf("Mode: 2 (OpenMP with collapse(2))\n");
    printf("Threads: %d\n", omp_get_max_threads());
    printf("Kernel Time: %f s\n", kernel_time);
    printf("Total Time: %f s\n", total_time);
    printf("sumC: %f\n", sumC);
    printf("maxC: %f\n", maxC);
    printf("checksum: %lld\n", checksum);
  }
  else if (mode == 3) {
    // Mode 3: OpenMP Synchonization comparison with atomics and critical sections
    int i, j, k; // Declare loop variables for private scope
    #pragma omp parallel for default(none) shared(A, B, C, N) private(i, j, k)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        double temp = 0.0;
        for (k = 0; k < N; k++) {
          temp += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = temp;
      }
    }
    kernel_time = omp_get_wtime() - start_time;

    // Analytics with reductions for sum and max
    #pragma omp parallel for default(none) shared(C, N) private(i, j) reduction(+:sumC) reduction(max:maxC)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        double val = C[i * N + j];
        sumC += val;
        if (val > maxC) maxC = val;
      }
    }

    // Checksum for modes 3A and 3B
    #pragma omp parallel for default(none) shared(C, N, checksum) private(i, j)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        long long local_val = (long long)(C[i * N + j] * 1000.0) % 100000;
        // For Mode 3A, use atomic to update checksum
        #pragma omp atomic
        checksum += local_val;
        // For Mode 3B, use critical section to update checksum
        
        // #pragma omp critical
        // {
        //   checksum += local_val;
        // }
        
      }
    }
    total_time = omp_get_wtime() - start_time;

    printf("Mode: 3 (OpenMP Synchronization (Atomic vs Critical in checksum))\n");
    printf("Threads: %d\n", omp_get_max_threads());
    printf("Kernel Time: %f s\n", kernel_time);
    printf("Total Time: %f s\n", total_time);
    printf("sumC: %f\n", sumC);
    printf("maxC: %f\n", maxC);
    printf("checksum: %lld\n", checksum);
  }
  else if (mode == 4) {
    // Mode 4: Task based parallelism with OpenMP tasks
    #pragma omp parallel 
    {
      // Single thread creates tasks for matrix multiplication
      #pragma omp single 
      {
        // Taskgroup to ensure all tasks are completed before proceeding to analytics
        #pragma omp taskgroup 
        {
          for (int row = 0; row < N; row++) {
            // Create a task for each row of the result matrix C
            #pragma omp task firstprivate(row) shared(A, B, C, N) 
            {
              for (int col = 0; col < N; col++) {
                double temp = 0.0;
                for (int k = 0; k < N; k++) {
                  temp += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = temp;
              }
            }
          }
        } // End of taskgroup, ensures all multiplication tasks are completed
      } 
    }
    kernel_time = omp_get_wtime() - start_time;

    // Analytics with reductions for sum and max
    int i, j; // Declare loop variables for private scope
    #pragma omp parallel for default(none) shared(C, N) private(i, j) reduction(+:sumC) reduction(max:maxC)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        double val = C[i * N + j];
        sumC += val;
        if (val > maxC) maxC = val;
      }
    }

    // Checksum using atomic for consistency for Mode 4
    #pragma omp parallel for default(none) shared(C, N, checksum) private(i, j)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        long long local_val = (long long)(C[i * N + j] * 1000.0) % 100000;
        #pragma omp atomic
        checksum += local_val;
      }
    }
    total_time = omp_get_wtime() - start_time;

    printf("Mode: 4 (OpenMP Task-based Parallelism)\n");
    printf("Threads: %d\n", omp_get_max_threads());
    printf("Kernel Time: %f s\n", kernel_time);
    printf("Total Time: %f s\n", total_time);
    printf("sumC: %f\n", sumC);
    printf("maxC: %f\n", maxC);
    printf("checksum: %lld\n", checksum);
  }
  else if (mode == 5) {
    // Mode 5: SIMD vectorization with single thread
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        double temp = 0.0;
        // Using OpenMP simd directive to vectorize the innermost loop
        #pragma omp simd reduction(+:temp)
        for (int k = 0; k < N; k++) {
          temp += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = temp;
      }
    }
    kernel_time = omp_get_wtime() - start_time;

    // SIMD analytics with reductions for sum and max
    #pragma omp simd reduction(+:sumC) reduction(max:maxC)
    for (int x = 0; x < N * N; x++) {
      double val = C[x];
      sumC += val;
      if (val > maxC) maxC = val;
    }

    // SIMD checksum
    #pragma omp simd reduction(+:checksum)
    for (int x = 0; x < N * N; x++) {
      checksum += (long long)(C[x] * 1000.0) % 100000;
    }
    total_time = omp_get_wtime() - start_time;

    printf("Mode: 5 (SIMD Vectorization)\n");
    printf("Threads: 1 (SIMD)\n");
    printf("Kernel Time: %f s\n", kernel_time);
    printf("Total Time: %f s\n", total_time);
    printf("sumC: %f\n", sumC);
    printf("maxC: %f\n", maxC);
    printf("checksum: %lld\n", checksum);
  }
  else if (mode == 6) {
    // Mode 6: Threads + SIMD
    int i, j, k;
    #pragma omp parallel for default(none) shared(A, B, C, N) private(i, j, k)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        double temp = 0.0;
        #pragma omp simd reduction(+:temp)
        for (k = 0; k < N; k++) {
          temp += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = temp;
      }
    }
    kernel_time = omp_get_wtime() - start_time;

    // Analytics with reductions for sum and max
    #pragma omp parallel for simd reduction(+:sumC) reduction(max:maxC)
    for (int x = 0; x < N * N; x++) {
      double val = C[x];
      sumC += val;
      if (val > maxC) maxC = val;
    }

    // SIMD checksum with reduction
    #pragma omp parallel for simd reduction(+:checksum)
    for (int x = 0; x < N * N; x++) {
      checksum += (long long)(C[x] * 1000.0) % 100000;
    }
    total_time = omp_get_wtime() - start_time;

    printf("Mode: 6 (Threads + SIMD)\n");
    printf("Threads: %d\n", omp_get_max_threads());
    printf("Kernel Time: %f s\n", kernel_time);
    printf("Total Time: %f s\n", total_time);
    printf("sumC: %f\n", sumC);
    printf("maxC: %f\n", maxC);
    printf("checksum: %lld\n", checksum);
  }

  // Free allocated memory
  free(A);
  free(B);
  free(C);

  return 0;
}