#ifndef QRNN_FO_POOL_CPU_H
#define QRNN_FO_POOL_CPU_H

#include "fo_pool_op.h"
#include "thread_pool.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/framework/types.h"

TF_QRNN_NAMESPACE_BEGIN
TF_QRNN_FO_POOL_NAMESPACE_BEGIN

typedef tensorflow::int64 int64;

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

template<typename FT>
void time_major_fo_pool(tensorflow::OpKernelContext* context,
             FT *dst, const FT *x, const FT *f, const FT *initial_state, int SEQ, int batch_size, int HIDDEN) {
    /*
    Note: destination is assumed to be one timestep longer than f or x where dst[0] = h_{-1}
    This means dst array has a separate index than that of f or x
    */
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    const tensorflow::int64 cost = SEQ * HIDDEN * 1000;
    Shard(worker_threads.num_threads, worker_threads.num_threads, batch_size, cost,
         [&batch_size, x, f, initial_state, dst, &HIDDEN, &SEQ](const int start, const int limit) {
        for (int batch_id = start; batch_id < limit; ++batch_id) {
            for (int hid = 0; hid < HIDDEN; hid++) {
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
                    dst[dst_i] = f[i] * x[i];
                    dst[dst_i] += (1 - f[i]) * dst[dst_iminus1];
                }
            }
        }
    });
}

template<typename FT>
void batch_major_fo_pool(tensorflow::OpKernelContext* context,
             FT *dst, const FT *x, const FT *f, const FT *initial_state, int SEQ, int batch_size, int HIDDEN) {
    /*
    Note: destination is assumed to be one timestep longer than f or x where dst[0] = h_{-1}
    This means dst array has a separate index than that of f or x
    */
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    const tensorflow::int64 cost = SEQ * HIDDEN * 1000;
    Shard(worker_threads.num_threads, worker_threads.num_threads, batch_size, cost,
         [&batch_size, x, f, initial_state, dst, &HIDDEN, &SEQ](const int start, const int limit) {
        for (int batch_id = start; batch_id < limit; ++batch_id) {
            for (int hid = 0; hid < HIDDEN; hid++) {
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
                    dst[dst_i] = f[i] * x[i];
                    dst[dst_i] += (1 - f[i]) * dst[dst_iminus1];
                }
            }
        }
    });
}

template<typename FT>
void time_major_bwd_fo_pool(tensorflow::OpKernelContext* context,
                 const FT *h, const FT *x, const FT *f, const FT *gh, FT *gx, FT *gf, FT *ginitial_state,
                 int SEQ, int batch_size, int HIDDEN) {
    /*
    Note: h is assumed to be one timestep longer than f, x, gf, gx, or gh where dst[0] = h_{-1}
    This means dst array has a separate index than that of f or x
    */
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    const tensorflow::int64 cost = SEQ * HIDDEN * 1000;
    Shard(worker_threads.num_threads, worker_threads.num_threads,
          batch_size, cost, [&batch_size, h, f, x, gh, gf, gx, ginitial_state, &HIDDEN, &SEQ](const int start, const int limit) {
        for (int batch_id = start; batch_id < limit; ++batch_id) {
            for (int hid = 0; hid < HIDDEN; hid++) {
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
                    // The line below is likely more numerically stable than (1 - f[i]) * running_f;
                    running_f       = running_f - f[i] * running_f;
                }
                ginitial_state[batch_id * HIDDEN + hid] = running_f + gh[batch_id * HIDDEN + hid];
            }
        }
    });
}

template<typename FT>
void batch_major_bwd_fo_pool(tensorflow::OpKernelContext* context,
                 const FT *h, const FT *x, const FT *f, const FT *gh, FT *gx, FT *gf, FT *ginitial_state,
                 int SEQ, int batch_size, int HIDDEN) {
    /*
    Note: h is assumed to be one timestep longer than f, x, gf, gx, or gh where dst[0] = h_{-1}
    This means dst array has a separate index than that of f or x
    */
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    const tensorflow::int64 cost = SEQ * HIDDEN * 1000;
    Shard(worker_threads.num_threads, worker_threads.num_threads,
          batch_size, cost, [&batch_size, h, f, x, gh, gf, gx, ginitial_state, &HIDDEN, &SEQ](const int start, const int limit) {
        for (int batch_id = start; batch_id < limit; ++batch_id) {
            for (int hid = 0; hid < HIDDEN; hid++) {
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
                    // The line below is likely more numerically stable than (1 - f[i]) * running_f;
                    running_f       = running_f - f[i] * running_f;
                }
                ginitial_state[batch_id * HIDDEN + hid] = running_f + gh[batch_id * HIDDEN * (SEQ + 1) + hid];
            }
        }
    });
}

// Specialise the FoPool op for CPUs
template <typename FT, bool time_major>
class FoPool<CPUDevice, FT, time_major> : public tensorflow::OpKernel {
    public:
        explicit FoPool(tensorflow::OpKernelConstruction * context) :
            tensorflow::OpKernel(context) {}

        void Compute(tensorflow::OpKernelContext * context) override {
            namespace tf = tensorflow;

            // Create reference to input Tensorflow tensors
            const auto & in_x = context->input(0);
            const auto & in_forget = context->input(1);
            const auto & in_initial_state = context->input(2);


            // Extract Eigen tensors
            auto x = in_x.flat<FT>().data();
            auto forget = in_forget.flat<FT>().data();
            auto initial_state = in_initial_state.flat<FT>().data();

            // Allocate output tensors
            // Allocate space for output tensor 'output'
            tf::Tensor * output_ptr = nullptr;
            auto in_x_shape = in_x.shape();
            tf::TensorShape output_shape = in_x_shape;
            if (time_major) {
                output_shape.set_dim(0, output_shape.dim_size(0) + 1);
            } else {
                output_shape.set_dim(1, output_shape.dim_size(1) + 1);
            }
            OP_REQUIRES_OK(context, context->allocate_output(
                0, output_shape, &output_ptr));
            auto out = output_ptr->flat<FT>().data();
            if (time_major) {
                time_major_fo_pool(context,
                                   out,
                                   x,
                                   forget,
                                   initial_state,
                                   in_x_shape.dim_size(0),
                                   output_shape.dim_size(1),
                                   output_shape.dim_size(2));
            } else {
                batch_major_fo_pool(context,
                                    out,
                                    x,
                                    forget,
                                    initial_state,
                                    in_x_shape.dim_size(1),
                                    output_shape.dim_size(0),
                                    output_shape.dim_size(2));
            }
        }
};

template <typename FT, bool time_major>
class BwdFoPool<CPUDevice, FT, time_major> : public tensorflow::OpKernel {
    public:
        explicit BwdFoPool(tensorflow::OpKernelConstruction * context) :
            tensorflow::OpKernel(context) {}

        void Compute(tensorflow::OpKernelContext * context) override {
            namespace tf = tensorflow;

            const auto& in_h = context->input(0);
            const auto& in_x = context->input(1);
            const auto& in_forget = context->input(2);
            const auto& in_gh = context->input(3);

            // Extract Eigen tensors
            auto h = in_h.flat<FT>().data();
            auto x = in_x.flat<FT>().data();
            auto forget = in_forget.flat<FT>().data();
            auto gh = in_gh.flat<FT>().data();

            // Allocate output tensors
            // Allocate space for output tensor 'output'
            tf::Tensor * out_gx = nullptr;
            tf::Tensor * out_gf = nullptr;
            tf::Tensor * out_ginitial_state = nullptr;

            auto in_x_shape = in_x.shape();
            tf::TensorShape grad_shape = in_x_shape;
            int batch_size = time_major ? in_x_shape.dim_size(1) : in_x_shape.dim_size(0);
            tf::TensorShape ginitial_state_shape({batch_size,
                                                  in_x_shape.dim_size(2)});

            OP_REQUIRES_OK(context, context->allocate_output(
                0, grad_shape, &out_gx));
            OP_REQUIRES_OK(context, context->allocate_output(
                1, grad_shape, &out_gf));
            OP_REQUIRES_OK(context, context->allocate_output(
                2, ginitial_state_shape, &out_ginitial_state));
            auto gx = out_gx->flat<FT>().data();
            auto gf = out_gf->flat<FT>().data();
            auto ginitial_state = out_ginitial_state->flat<FT>().data();

            if (time_major) {
                time_major_bwd_fo_pool(context,
                                       h,
                                       x,
                                       forget,
                                       gh,
                                       gx,
                                       gf,
                                       ginitial_state,
                                       grad_shape.dim_size(0),
                                       grad_shape.dim_size(1),
                                       grad_shape.dim_size(2));
            } else {
                batch_major_bwd_fo_pool(context,
                                        h,
                                        x,
                                        forget,
                                        gh,
                                        gx,
                                        gf,
                                        ginitial_state,
                                        grad_shape.dim_size(1),
                                        grad_shape.dim_size(0),
                                        grad_shape.dim_size(2));
            }
        }
};

TF_QRNN_FO_POOL_NAMESPACE_STOP
TF_QRNN_NAMESPACE_STOP

#endif // #ifndef QRNN_FO_POOL_OP_CPU_H
