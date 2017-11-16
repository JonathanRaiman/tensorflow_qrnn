#include "fo_pool_op_kernel.h"

struct KernelParams {
    dim3 grid;
    dim3 blocks;
    KernelParams(int HIDDEN, int batch_size) :
        grid(std::ceil(double(HIDDEN / double(min(HIDDEN, 512)))), batch_size, 1),
        blocks(min(HIDDEN, 512), 1, 1) {};
};


/* TIME MAJOR */

template<typename FT>
__global__
void time_major_fo_pool(FT *dst, const FT *x, const FT *f, const FT *initial_state, int SEQ, int batch_size, int HIDDEN) {
    /*
    Note: destination is assumed to be one timestep longer than f or x where dst[0] = h_{-1}
    This means dst array has a separate index than that of f or x
    */
    int hid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y * blockDim.y + threadIdx.y;
    if (hid >= HIDDEN || batch_id >= batch_size)
        return;
    //
    dst[batch_id * HIDDEN + hid] = initial_state[batch_id * HIDDEN + hid];
    for (int ts = 0 + 1; ts < SEQ + 1; ts++) {
        // Good sanity check for debugging - only perform additions to a zeroed chunk of memory
        // Addition seems atomic or near atomic - you should get incorrect answers if doubling up via threads
        // Note: the index i needs to be offset by one as f[0] (f_t) is used for dst[1] (h_t) etc
        // To move timesteps, we step HIDDEN * batch_size
        // To move batches, we move HIDDEN
        // To move neurons, we move +- 1
        // Note: dst[dst_i] = ts * 100 + batch_id * 10 + hid; is useful for debugging
        int i           = (ts - 1) * HIDDEN * batch_size + batch_id * HIDDEN + hid;
        int dst_i       = (ts - 0) * HIDDEN * batch_size + batch_id * HIDDEN + hid;
        int dst_iminus1 = (ts - 1) * HIDDEN * batch_size + batch_id * HIDDEN + hid;
        dst[dst_i]      = f[i] * x[i];
        dst[dst_i]      += (1 - f[i]) * dst[dst_iminus1];
    }
}

template<typename FT>
__global__
void time_major_bwd_fo_pool(const FT *h, const FT *x, const FT *f, const FT *gh, FT *gx, FT *gf, FT *ginitial_state, int SEQ, int batch_size, int HIDDEN) {
    /*
    Note: h is assumed to be one timestep longer than f, x, gf, gx, or gh where dst[0] = h_{-1}
    This means dst array has a separate index than that of f or x
    */
    int hid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y * blockDim.y + threadIdx.y;
    if (hid >= HIDDEN || batch_id >= batch_size)
        return;
    //
    double running_f = 0;
    for (int ts = SEQ - 1 + 1; ts >= 0 + 1; ts--) {
        int i           = (ts - 1) * HIDDEN * batch_size + batch_id * HIDDEN + hid;
        int dst_iminus1 = (ts - 1) * HIDDEN * batch_size + batch_id * HIDDEN + hid;
        //
        running_f       += gh[dst_iminus1];
        // Gradient of X
        gx[i]           = f[i] * running_f;
        // Gradient of F
        gf[i]           = (x[i] - h[dst_iminus1]) * running_f;
        //
        // The line below is likely more numerically stable than (1 - f[i]) * running_f;
        running_f       = running_f - f[i] * running_f;
    }
    ginitial_state[batch_id * HIDDEN + hid] = running_f + gh[batch_id * HIDDEN + hid];
}

void TimeMajorFoPoolLauncher(float *dst, const float *x, const float *f, const float *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream) {
    KernelParams l(HIDDEN, batch_size);
    time_major_fo_pool<<<l.grid, l.blocks, 0, stream>>>(dst, x, f, initial_state, SEQ, batch_size, HIDDEN);
}
void TimeMajorFoPoolLauncher(double *dst, const double *x, const double *f, const double *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream) {
    KernelParams l(HIDDEN, batch_size);
    time_major_fo_pool<<<l.grid, l.blocks, 0, stream>>>(dst, x, f, initial_state, SEQ, batch_size, HIDDEN);
}
void TimeMajorBwdFoPoolLauncher(const float *h, const float *x, const float *f, const float *gh, float *gx, float *gf, float *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream) {
    KernelParams l(HIDDEN, batch_size);
    time_major_bwd_fo_pool<<<l.grid, l.blocks, 0, stream>>>(h, x, f, gh, gx, gf, ginitial_state, SEQ, batch_size, HIDDEN);
}
void TimeMajorBwdFoPoolLauncher(const double *h, const double *x, const double *f, const double *gh, double *gx, double *gf, double *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream) {
    KernelParams l(HIDDEN, batch_size);
    time_major_bwd_fo_pool<<<l.grid, l.blocks, 0, stream>>>(h, x, f, gh, gx, gf, ginitial_state, SEQ, batch_size, HIDDEN);
}


/* BATCH MAJOR */

template<typename FT>
__global__
void batch_major_fo_pool(FT *dst, const FT *x, const FT *f, const FT *initial_state, int SEQ, int batch_size, int HIDDEN) {
    /*
    Note: destination is assumed to be one timestep longer than f or x where dst[0] = h_{-1}
    This means dst array has a separate index than that of f or x
    */
    int hid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y * blockDim.y + threadIdx.y;
    if (hid >= HIDDEN || batch_id >= batch_size)
        return;
    //
    dst[batch_id * HIDDEN * (SEQ + 1) + hid] = initial_state[batch_id * HIDDEN + hid];
    for (int ts = 0 + 1; ts < SEQ + 1; ts++) {
        // Good sanity check for debugging - only perform additions to a zeroed chunk of memory
        // Addition seems atomic or near atomic - you should get incorrect answers if doubling up via threads
        // Note: the index i needs to be offset by one as f[0] (f_t) is used for dst[1] (h_t) etc
        // To move timesteps, we step HIDDEN * batch_size
        // To move batches, we move HIDDEN
        // To move neurons, we move +- 1
        // Note: dst[dst_i] = ts * 100 + batch_id * 10 + hid; is useful for debugging
        int i           = (ts - 1) * HIDDEN + batch_id * HIDDEN * SEQ + hid;
        int dst_i       = (ts - 0) * HIDDEN + batch_id * HIDDEN * (SEQ + 1) + hid;
        int dst_iminus1 = (ts - 1) * HIDDEN + batch_id * HIDDEN * (SEQ + 1) + hid;
        dst[dst_i]      = f[i] * x[i];
        dst[dst_i]      += (1 - f[i]) * dst[dst_iminus1];
    }
}

template<typename FT>
__global__
void batch_major_bwd_fo_pool(const FT *h, const FT *x, const FT *f, const FT *gh, FT *gx, FT *gf, FT *ginitial_state, int SEQ, int batch_size, int HIDDEN) {
    /*
    Note: h is assumed to be one timestep longer than f, x, gf, gx, or gh where dst[0] = h_{-1}
    This means dst array has a separate index than that of f or x
    */
    int hid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y * blockDim.y + threadIdx.y;
    if (hid >= HIDDEN || batch_id >= batch_size)
        return;
    //
    double running_f = 0;
    for (int ts = SEQ - 1 + 1; ts >= 0 + 1; ts--) {
        int i           = (ts - 1) * HIDDEN + batch_id * HIDDEN * SEQ + hid;
        int dst_iminus1 = (ts - 1) * HIDDEN + batch_id * HIDDEN * (SEQ + 1) + hid;
        //
        running_f       += gh[dst_iminus1];
        // Gradient of X
        gx[i]           = f[i] * running_f;
        // Gradient of F
        gf[i]           = (x[i] - h[dst_iminus1]) * running_f;
        //
        // The line below is likely more numerically stable than (1 - f[i]) * running_f;
        running_f       = running_f - f[i] * running_f;
    }
    ginitial_state[batch_id * HIDDEN + hid] = running_f + gh[batch_id * HIDDEN * (SEQ + 1) + hid];
}


void BatchMajorFoPoolLauncher(float *dst, const float *x, const float *f, const float *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream) {
    KernelParams l(HIDDEN, batch_size);
    batch_major_fo_pool<<<l.grid, l.blocks, 0, stream>>>(dst, x, f, initial_state, SEQ, batch_size, HIDDEN);
}
void BatchMajorFoPoolLauncher(double *dst, const double *x, const double *f, const double *initial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream) {
    KernelParams l(HIDDEN, batch_size);
    batch_major_fo_pool<<<l.grid, l.blocks, 0, stream>>>(dst, x, f, initial_state, SEQ, batch_size, HIDDEN);
}
void BatchMajorBwdFoPoolLauncher(const float *h, const float *x, const float *f, const float *gh, float *gx, float *gf, float *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream) {
    KernelParams l(HIDDEN, batch_size);
    batch_major_bwd_fo_pool<<<l.grid, l.blocks, 0, stream>>>(h, x, f, gh, gx, gf, ginitial_state, SEQ, batch_size, HIDDEN);
}
void BatchMajorBwdFoPoolLauncher(const double *h, const double *x, const double *f, const double *gh, double *gx, double *gf, double *ginitial_state, int SEQ, int batch_size, int HIDDEN, cudaStream_t stream) {
    KernelParams l(HIDDEN, batch_size);
    batch_major_bwd_fo_pool<<<l.grid, l.blocks, 0, stream>>>(h, x, f, gh, gx, gf, ginitial_state, SEQ, batch_size, HIDDEN);
}
