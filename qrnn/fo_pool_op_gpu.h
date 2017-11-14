#if GOOGLE_CUDA

#ifndef QRNN_FO_POOL_OP_GPU_CUH
#define QRNN_FO_POOL_OP_GPU_CUH

#include "fo_pool_op.h"
#include "fo_pool_op_kernel.h"

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

TF_QRNN_NAMESPACE_BEGIN
TF_QRNN_FO_POOL_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

// Specialise the FoPool op for GPUs
template <typename FT>
class FoPool<GPUDevice, FT> : public tensorflow::OpKernel {
public:
    explicit FoPool(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override {
        namespace tf = tensorflow;

        // Create variables for input tensors
        const auto & in_x = context->input(0);
        const auto & in_forget = context->input(1);
        const auto & in_hinit = context->input(2);

        // Allocate output tensors
        // Allocate space for output tensor 'output'
        tf::Tensor * output_ptr = nullptr;
        auto in_x_shape = in_x.shape();
        tf::TensorShape output_shape = in_x_shape;
        output_shape.set_dim(0, output_shape.dim_size(0) + 1);
        OP_REQUIRES_OK(context, context->allocate_output(
            0, output_shape, &output_ptr));

        // Get pointers to flattened tensor data buffers
        const auto fin_x = in_x.flat<FT>().data();
        const auto fin_forget = in_forget.flat<FT>().data();
        const auto fin_hinit = in_hinit.flat<FT>().data();
        auto fout_output = output_ptr->flat<FT>().data();


        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Call the qrnn_fo_pool CUDA kernel
        FoPoolLauncher(fout_output, fin_x, fin_forget, fin_hinit,
                       in_x_shape.dim_size(0),
                       output_shape.dim_size(1),
                       output_shape.dim_size(2),
                       device.stream());
    }
};

template <typename FT>
class BwdFoPool<GPUDevice, FT> : public tensorflow::OpKernel {
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
        tf::Tensor * out_gf = nullptr;
        tf::Tensor * out_gx = nullptr;
        tf::Tensor * out_ginitial_state = nullptr;

        auto in_x_shape = in_x.shape();
        tf::TensorShape grad_shape = in_x_shape;
        tf::TensorShape ginitial_state_shape({in_x_shape.dim_size(1),
                                              in_x_shape.dim_size(2)});

        OP_REQUIRES_OK(context, context->allocate_output(
            0, grad_shape, &out_gx));
        OP_REQUIRES_OK(context, context->allocate_output(
            1, grad_shape, &out_gf));
        OP_REQUIRES_OK(context, context->allocate_output(
            2, ginitial_state_shape, &out_ginitial_state));
        auto gf = out_gf->flat<FT>().data();
        auto gx = out_gx->flat<FT>().data();
        auto ginitial_state = out_ginitial_state->flat<FT>().data();

        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        BwdFoPoolLauncher(h,
                          x,
                          forget,
                          gh,
                          gx,
                          gf,
                          ginitial_state,
                          grad_shape.dim_size(0),
                          grad_shape.dim_size(1),
                          grad_shape.dim_size(2),
                          device.stream());
    }
};

TF_QRNN_FO_POOL_NAMESPACE_STOP
TF_QRNN_NAMESPACE_STOP

#endif // #ifndef QRNN_FO_POOL_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
