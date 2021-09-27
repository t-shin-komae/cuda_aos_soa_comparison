#include "functions.h"
#include <math.h>
#include <stdio.h>

void sync_and_check_error() {
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Cuda error occurs: %s\n", cudaGetErrorString(err));
  }
}

__global__ void kernel_squareSoA(struct VectorNArray2DSoA src,
                                 struct VectorNArray2DSoA dst) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int Nx = src.Nx;
  int Ny = src.Ny;
  if (i < Nx && j < Ny) {
    for (int e = 0; e < N; e++) {
      double tmp = src.data[e][i + j * Nx];
      dst.data[e][i + j * Nx] = tmp * tmp;
    }
  } else {
    return;
  }
}
__global__ void kernel_squareAoS(const struct VectorN *src, struct VectorN *dst,
                                 int Nx, int Ny) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < Nx && j < Ny) {
    for (int e = 0; e < N; e++) {
      double tmp = src[i + j * Nx].data[e];
      dst[i + j * Nx].data[e] = tmp * tmp;
    }
  } else {
    return;
  }
}

__global__ void kernel_root_mean_squareSoA(struct VectorNArray2DSoA src,
                                           double *dst) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int Nx = src.Nx;
  int Ny = src.Ny;
  if (i < Nx && j < Ny) {
    double sum = 0.0;
    for (int e = 0; e < N; e++) {
      double tmp = src.data[e][i + j * Nx];
      sum += tmp * tmp;
    }
    dst[i + j * Nx] = sqrt(sum/N);
  } else {
    return;
  }
}
__global__ void kernel_root_mean_squareAoS(struct VectorN *src, double *dst,
                                           int Nx, int Ny) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < Nx && j < Ny) {
    double sum = 0.0;
    for (int e = 0; e < N; e++) {
      double tmp = src[i + j * Nx].data[e];
      sum += tmp * tmp;
    }
    dst[i + j * Nx] = sqrt(sum/N);
  } else {
    return;
  }
}

__global__ void kernel_multiply_matrixSoA(struct VectorNArray2DSoA src,
                                          struct VectorNArray2DSoA dst,
                                          struct MatrixNN mat) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int Nx = src.Nx;
  int Ny = src.Ny;
  if (i < Nx && j < Ny) {
    for (int k = 0; k < N; k++)
      for (int l = 0; l < N; l++)
        dst.data[k][i + j * Nx] = mat.data[k][l] * src.data[l][i + j * Nx];
  } else {
    return;
  }
}
__global__ void kernel_multiply_matrixAoS(const struct VectorN *src,
                                          struct VectorN *dst,
                                          struct MatrixNN mat, int Nx, int Ny) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < Nx && j < Ny) {
    for (int k = 0; k < N; k++)
      for (int l = 0; l < N; l++)
        dst[i + j * Nx].data[k] = mat.data[k][l] * src[i + j * Nx].data[l];
  } else {
    return;
  }
}

__global__ void kernel_debugSoA(struct VectorNArray2DSoA d_soa) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int Nx = d_soa.Nx;
  int Ny = d_soa.Ny;
  if (i < Nx && j < Ny) {
    for (int e = 0; e < N; e++) {
      printf("(%d,%d)[%d]=%lf\n", i, j, e, d_soa.data[e][i + j * Nx]);
    }
  } else {
    return;
  }
}
__global__ void kernel_debugAoS(struct VectorN *d_aos, int Nx, int Ny) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < Nx && j < Ny) {
    for (int e = 0; e < N; e++) {
      printf("(%d,%d)[%d]=%lf\n", i, j, e, d_aos[i + j * Nx].data[e]);
    }
  } else {
    return;
  }
}

// -----------------** HOST CODE **----------------------
void squareSoA(struct VectorNArray2DSoA src, struct VectorNArray2DSoA dst) {
  kernel_squareSoA<<<blocks, threads>>>(src, dst);
  sync_and_check_error();
}
void squareAoS(const struct VectorN *src, struct VectorN *dst, int Nx, int Ny) {
  kernel_squareAoS<<<blocks, threads>>>(src, dst, Nx, Ny);
  sync_and_check_error();
}
void root_mean_squareSoA(struct VectorNArray2DSoA src, double *dst) {
  kernel_root_mean_squareSoA<<<blocks, threads>>>(src, dst);
  sync_and_check_error();
}
void root_mean_squareAoS(struct VectorN *src, double *dst, int Nx, int Ny) {
  kernel_root_mean_squareAoS<<<blocks, threads>>>(src, dst, Nx, Ny);
  sync_and_check_error();
}
void multiply_matrixSoA(struct VectorNArray2DSoA src,
                        struct VectorNArray2DSoA dst, struct MatrixNN mat) {
  kernel_multiply_matrixSoA<<<blocks, threads>>>(src, dst, mat);
  sync_and_check_error();
}
void multiply_matrixAoS(const struct VectorN *src, struct VectorN *dst,
                        struct MatrixNN mat, int Nx, int Ny) {
  kernel_multiply_matrixAoS<<<blocks, threads>>>(src, dst, mat, Nx, Ny);
  sync_and_check_error();
}
void debugSoA(struct VectorNArray2DSoA d_soa) {
  kernel_debugSoA<<<blocks, threads>>>(d_soa);
  sync_and_check_error();
}
void debugAoS(struct VectorN *d_aos, int Nx, int Ny) {
  kernel_debugAoS<<<blocks, threads>>>(d_aos, Nx, Ny);
  sync_and_check_error();
}
// ---------------** HOST CODE END **--------------------
