#if GOOGLE_CUDA

#ifndef QRNN_RECURRENT_FORGET_MULT_OP_GPU_CUH
#define QRNN_RECURRENT_FORGET_MULT_OP_GPU_CUH

#include "recurrent_forget_mult_op.h"
#include "recurrent_forget_mult_op_kernel.h"

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

TF_QRNN_NAMESPACE_BEGIN
TF_QRNN_RECURRENT_FORGET_MULT_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

// Specialise the RecurrentForgetMult op for GPUs
template <typename FT>
class RecurrentForgetMult<GPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit RecurrentForgetMult(tensorflow::OpKernelConstruction * context) :
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
        tf::TensorShape output_shape = in_x.shape();
        OP_REQUIRES_OK(context, context->allocate_output(
            0, output_shape, &output_ptr));

        // Get pointers to flattened tensor data buffers
        const auto fin_x = in_x.flat<FT>().data();
        const auto fin_forget = in_forget.flat<FT>().data();
        auto fout_output = output_ptr->flat<FT>().data();


        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Call the qrnn_recurrent_forget_mult CUDA kernel
        RecurrentForgetMultLauncher(fout_output, fin_forget, fin_x,
                                    output_shape.dim_size(0),
                                    output_shape.dim_size(1),
                                    output_shape.dim_size(2),
                                    device.stream());
    }
};

TF_QRNN_RECURRENT_FORGET_MULT_NAMESPACE_STOP
TF_QRNN_NAMESPACE_STOP

#endif // #ifndef QRNN_RECURRENT_FORGET_MULT_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
