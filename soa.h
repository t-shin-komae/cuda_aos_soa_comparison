#pragma once
#include <cuda_runtime.h>

const int N = 6;

struct VectorNArray2DSoA {
  double *data[N];
  int Nx, Ny;
};
struct VectorN {
  double data[N];
};

struct VectorNArray2DSoA makeGPUArray2DSoA(int Nx, int Ny);
struct VectorNArray2DSoA makeCPUArray2DSoA(int Nx, int Ny);

void DeleteGPUArray2DSoA(struct VectorNArray2DSoA *vecarray);
void DeleteCPUArray2DSoA(struct VectorNArray2DSoA *vecarray);

void copyArray2DSoA(struct VectorNArray2DSoA *src, struct VectorNArray2DSoA *dst,
                 cudaMemcpyKind kind);
void copyArray2DAoS(struct VectorN* src, struct VectorN *dst, int Nx, int Ny,
                 cudaMemcpyKind kind);

struct VectorN *makeArray2DAoS(int Nx, int Ny);
void DeleteArray2DAoS(struct VectorN *);
