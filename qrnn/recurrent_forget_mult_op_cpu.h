#ifndef QRNN_RECURRENT_FORGET_MULT_OP_CPU_H
#define QRNN_RECURRENT_FORGET_MULT_OP_CPU_H

#include "recurrent_forget_mult_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

TF_QRNN_NAMESPACE_BEGIN
TF_QRNN_RECURRENT_FORGET_MULT_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the RecurrentForgetMult op for CPUs
template <typename FT>
class RecurrentForgetMult<CPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit RecurrentForgetMult(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create reference to input Tensorflow tensors
        const auto & in_x = context->input(0);
        const auto & in_forget = context->input(1);


        // Extract Eigen tensors
        auto x = in_x.tensor<FT, 3>();
        auto forget = in_forget.tensor<FT, 3>();


        // Allocate output tensors
        // Allocate space for output tensor 'output'
        tf::Tensor * output_ptr = nullptr;
        tf::TensorShape output_shape = tf::TensorShape({ 1, 1, 1 });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, output_shape, &output_ptr));

    }
};

TF_QRNN_RECURRENT_FORGET_MULT_NAMESPACE_STOP
TF_QRNN_NAMESPACE_STOP

#endif // #ifndef QRNN_RECURRENT_FORGET_MULT_OP_CPU_H
