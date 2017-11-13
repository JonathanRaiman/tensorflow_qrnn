#ifndef RECURRENT_FORGET_MULT_OP_KERNEL_H
#define RECURRENT_FORGET_MULT_OP_KERNEL_H

#include <cuda_runtime.h>

void RecurrentForgetMultLauncher(float *dst, const float *f, const float *x, int SEQ, int BATCH, int HIDDEN);
void BwdRecurrentForgetMultLauncher(const float *h, const float *f, const float *x, const float *gh, float *gf, float *gx, float *ghinit, int SEQ, int BATCH, int HIDDEN);

#endif
