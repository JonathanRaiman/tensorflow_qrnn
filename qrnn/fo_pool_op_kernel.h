#ifndef RECURRENT_FORGET_MULT_OP_KERNEL_H
#define RECURRENT_FORGET_MULT_OP_KERNEL_H

#include <cuda_runtime.h>

void FoPoolLauncher(float *dst, const float *f, const float *x, const float *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream);
void BwdFoPoolLauncher(const float *h, const float *f, const float *x, const float *gh, float *gf, float *gx, float *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream);

#endif
