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



  // Free allocated memory
  free(A);
  free(B);
  free(C);

  return 0;
}