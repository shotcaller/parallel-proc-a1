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
    
  }

  // Free allocated memory
  free(A);
  free(B);
  free(C);

  return 0;
}