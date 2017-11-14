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
class FoPool<GPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit FoPool(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create variables for input tensors
        const auto & in_x = context->input(0);
        const auto & in_forget = context->input(1);

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
        auto fout_output = output_ptr->flat<FT>().data();


        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Call the qrnn_fo_pool CUDA kernel
        FoPoolLauncher(fout_output, fin_forget, fin_x,
                       in_x_shape.dim_size(0),
                       output_shape.dim_size(1),
                       output_shape.dim_size(2),
                       device.stream());
    }
};

TF_QRNN_FO_POOL_NAMESPACE_STOP
TF_QRNN_NAMESPACE_STOP

#endif // #ifndef QRNN_FO_POOL_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
