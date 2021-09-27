#include "soa.h"
#include <stdlib.h>
struct VectorNArray2DSoA makeGPUArray2DSoA(int Nx, int Ny) {
  struct VectorNArray2DSoA vecarray;
  vecarray.Nx = Nx;
  vecarray.Ny = Ny;
  for (int i = 0; i < N; i++) {
    cudaMalloc(&vecarray.data[i], sizeof(double) * Nx * Ny);
  }
  return vecarray;
}
struct VectorNArray2DSoA makeCPUArray2DSoA(int Nx, int Ny) {
  struct VectorNArray2DSoA vecarray;
  vecarray.Nx = Nx;
  vecarray.Ny = Ny;
  for (int i = 0; i < N; i++) {
    vecarray.data[i] = (double *)malloc(sizeof(double) * Nx * Ny);
  }
  return vecarray;
}
void copyArray2DSoA(struct VectorNArray2DSoA *src, struct VectorNArray2DSoA *dst,
        cudaMemcpyKind kind){
  for (int i = 0; i < N; i++) {
    cudaMemcpy(dst->data[i],src->data[i],sizeof(double)*src->Nx*src->Ny,kind);
  }
}
void copyArray2DAoS(struct VectorN* src, struct VectorN *dst, int Nx, int Ny,
        cudaMemcpyKind kind){
    cudaMemcpy(dst,src,sizeof(struct VectorN)* Nx*Ny,kind);
}

void DeleteGPUArray2DSoA(struct VectorNArray2DSoA *vecarray) {
  for (int i = 0; i < N; i++) {
    cudaFree(vecarray->data[i]);
  }
}
void DeleteCPUArray2DSoA(struct VectorNArray2DSoA *vecarray) {
  for (int i = 0; i < N; i++) {
    free(vecarray->data[i]);
  }
}

struct VectorN *makeArray2DAoS(int Nx, int Ny) {
  struct VectorN *ptr;
  cudaMalloc(&ptr, sizeof(struct VectorN) * Nx * Ny);
  return ptr;
}
void DeleteArray2DAoS(struct VectorN *ptr) { cudaFree(ptr); }
