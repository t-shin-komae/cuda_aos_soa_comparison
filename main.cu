#include "functions.h"
#include "soa.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

dim3 threads;
dim3 blocks;

long time_diff_us(struct timeval st, struct timeval et) {
  return (et.tv_sec - st.tv_sec) * 1000000 + (et.tv_usec - st.tv_usec);
}

void init_soa(struct VectorNArray2DSoA *h_soa) {
  int Nx = h_soa->Nx;
  int Ny = h_soa->Ny;
  for (int j = 0; j < Ny; j++) {
    for (int i = 0; i < Nx; i++) {
      for (int e = 0; e < N; e++) {
        h_soa->data[e][i + j * Nx] = e * (i + j * Nx);
      }
    }
  }
}
void init_aos(struct VectorN *h_aos, int Nx, int Ny) {
  for (int j = 0; j < Ny; j++) {
    for (int i = 0; i < Nx; i++) {
      for (int e = 0; e < N; e++) {
        h_aos[i + j * Nx].data[e] = e * (i + j * Nx);
      }
    }
  }
}

int main() {
  for (int Nx = 1 << 8; Nx <= 1 << 12; Nx <<= 4) {
    int Ny = Nx;
    printf("Vector size: %d, Nx = %d, Ny = %d\n", N, Nx, Ny);
    struct timeval st, et;
    long us;
    double *d_array;
    struct VectorNArray2DSoA h_soa, d_soa, d_soa_dst;
    struct VectorN *h_aos, *d_aos, *d_aos_dst;
    struct MatrixNN mat;

    // ---- set num threads and blocks -----
    threads = dim3(32, 32);                        // change global variable
    blocks = dim3((Nx + 31) / 32, (Ny + 31) / 32); // change global variable

    // ---- initialize mat --------
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        mat.data[i][j] = i + j * N;

    // ---- host SoA array initialization -----
    h_soa = makeCPUArray2DSoA(Nx, Ny);
    init_soa(&h_soa);

    // ---- host AoS array initialization -----
    h_aos = (struct VectorN *)malloc(sizeof(struct VectorN) * Nx * Ny);
    init_aos(h_aos, Nx, Ny);

    // ---- Allocate device array ------
    cudaMalloc(&d_array, sizeof(double) * Nx * Ny);

    // ---- Allocate device SoA array -----
    d_soa = makeGPUArray2DSoA(Nx, Ny);
    d_soa_dst = makeGPUArray2DSoA(Nx, Ny);
    copyArray2DSoA(&h_soa, &d_soa, cudaMemcpyHostToDevice);

    // ---- Allocate device AoS array -----
    d_aos = makeArray2DAoS(Nx, Ny);
    d_aos_dst = makeArray2DAoS(Nx, Ny);
    copyArray2DAoS(h_aos, d_aos, Nx, Ny, cudaMemcpyHostToDevice);

    // ====== Run SoA array operation ========
    {
      // ====== square ========
      // printf("default\n");
      // debugSoA(d_soa);
      gettimeofday(&st, NULL);
      squareSoA(d_soa, d_soa_dst);
      gettimeofday(&et, NULL);
      us = time_diff_us(st, et);
      // printf("result\n");
      // debugSoA(d_soa_dst);
      printf("squareSoA:%ld\n", us);
      // ====== root_mean_square ========
      gettimeofday(&st, NULL);
      root_mean_squareSoA(d_soa, d_array);
      gettimeofday(&et, NULL);
      us = time_diff_us(st, et);
      printf("root_mean_squareSoA:%ld\n", us);
      // ====== matrix mul ========
      gettimeofday(&st, NULL);
      multiply_matrixSoA(d_soa, d_soa_dst, mat);
      gettimeofday(&et, NULL);
      us = time_diff_us(st, et);
      printf("matrixmulSoA:%ld\n", us);
    }
    {
      gettimeofday(&st, NULL);
      squareAoS(d_aos, d_aos_dst, Nx, Ny);
      gettimeofday(&et, NULL);
      us = time_diff_us(st, et);
      printf("squareAoS:%ld\n", us);
      // ====== root_mean_square ========
      gettimeofday(&st, NULL);
      root_mean_squareAoS(d_aos, d_array, Nx, Ny);
      gettimeofday(&et, NULL);
      us = time_diff_us(st, et);
      printf("root_mean_squareAoS:%ld\n", us);
      // ====== matrix mul ========
      gettimeofday(&st, NULL);
      multiply_matrixAoS(d_aos, d_aos_dst, mat, Nx, Ny);
      gettimeofday(&et, NULL);
      us = time_diff_us(st, et);
      printf("matrixmulAoS:%ld\n", us);
    }

    DeleteCPUArray2DSoA(&h_soa);
    DeleteGPUArray2DSoA(&d_soa);
    DeleteGPUArray2DSoA(&d_soa_dst);
  }
}
