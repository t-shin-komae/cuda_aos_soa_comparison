#pragma once
#include "soa.h"

extern dim3 blocks, threads;

struct MatrixNN {
  double data[N][N];
};

// -----------------**KERNEL CODE**----------------------

__global__ void kernel_squareSoA(struct VectorNArray2DSoA src,
                                 struct VectorNArray2DSoA dst);
__global__ void kernel_squareAoS(const struct VectorN *src, struct VectorN *dst,
                                 int Nx, int Ny);
__global__ void kernel_root_mean_squareSoA(struct VectorNArray2DSoA src,
                                           double *dst);
__global__ void kernel_root_mean_squareAoS(struct VectorN *src, double *dst,
                                           int Nx, int Ny);

__global__ void kernel_multiply_matrixSoA(struct VectorNArray2DSoA src,
                                          struct VectorNArray2DSoA dst,
                                          struct MatrixNN mat);
__global__ void kernel_multiply_matrixAoS(const struct VectorN *src,
                                          struct VectorN *dst,
                                          struct MatrixNN mat, int Nx, int Ny);


__global__ void kernel_debugSoA(struct VectorNArray2DSoA d_soa);
__global__ void kernel_debugAoS(struct VectorN *d_aos, int Nx, int Ny);

// ---------------**KERNEL CODE END **-------------------

// -----------------** HOST CODE **----------------------

void sync_and_check_error();
void squareSoA(struct VectorNArray2DSoA src, struct VectorNArray2DSoA dst);
void squareAoS(const struct VectorN *src, struct VectorN *dst, int Nx, int Ny);
void root_mean_squareSoA(struct VectorNArray2DSoA src, double *dst);
void root_mean_squareAoS(struct VectorN *src, double *dst, int Nx, int Ny);

void multiply_matrixSoA(struct VectorNArray2DSoA src,
                        struct VectorNArray2DSoA dst, struct MatrixNN mat);
void multiply_matrixAoS(const struct VectorN *src, struct VectorN *dst,
                        struct MatrixNN mat, int Nx, int Ny);
void debugSoA(struct VectorNArray2DSoA d_soa);
void debugAoS(struct VectorN *d_aos, int Nx, int Ny);
// ---------------** HOST CODE END **--------------------
