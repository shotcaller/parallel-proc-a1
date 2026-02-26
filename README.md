# CS6025-FR01B Assignment 1

**Name:** Ruturaj Sushil Ghodke
**Student ID**: 3775789
**Date:** 26 February 2026

## 1. Compilation Instructions
The program is written in C and utilizes OpenMP for multi-threading and SIMD vectorization. To achieve optimal performance, use the following compilation command:

```bash
gcc -O3 -fopenmp a1.c -lm -o a1
```

Flags Explanation:
1. -O3: Highest level of optimization for speed.
2. -fopenmp: Enables OpenMP directives.
3. -lm: Links the math library.

## Example Run Commands
The executable requires two arguments: <N> (Matrix Size) and <Mode> (0-6). Use the OMP_NUM_THREADS environment variable to control the thread count for parallel modes.

**Serial Baseline**
```bash
./a1 1024 0
```

**Parallel Loops**
```bash
export OMP_NUM_THREADS=22
./a1 1024 1
```

**Parallel Tasks**
```bash
export OMP_NUM_THREADS=8
./a1 1024 4
```

**Threads + SIMD**
```bash
export OMP_NUM_THREADS=16
./a1 1024 6
```

## Machine Specifications

1. CPU Model: Intel Core Ultra 7 155H
2. Cores: 16
3. Logical Processors: 22
4. Base Speed: 3.80 GHz
5. RAM: 16 GB

## Matrix Sizes (N)
Performance data was collected for the following matrix sizes:

- N = 512
- N = 1024
- N = 2048
