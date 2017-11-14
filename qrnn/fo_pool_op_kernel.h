#ifndef RECURRENT_FORGET_MULT_OP_KERNEL_H
#define RECURRENT_FORGET_MULT_OP_KERNEL_H

#include <cuda_runtime.h>

void FoPoolLauncher(float *dst, const float *x, const float *f, const float *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream);
void BwdFoPoolLauncher(const float *h, const float *x, const float *f, const float *gh, float *gx, float *gf, float *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream);

void FoPoolLauncher(double *dst, const double *x, const double *f, const double *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream);
void BwdFoPoolLauncher(const double *h, const double *x, const double *f, const double *gh, double *gx, double *gf, double *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream);

#endif
