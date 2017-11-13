#if GOOGLE_CUDA

#include "recurrent_forget_mult_op_gpu.h"
#include "recurrent_forget_mult_op_kernel.h"

TF_QRNN_NAMESPACE_BEGIN
TF_QRNN_RECURRENT_FORGET_MULT_NAMESPACE_BEGIN


// Register a GPU kernel for RecurrentForgetMult
// handling permutation ['float']
REGISTER_KERNEL_BUILDER(
    Name("RecurrentForgetMult")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_GPU),
    RecurrentForgetMult<GPUDevice, float>);

TF_QRNN_RECURRENT_FORGET_MULT_NAMESPACE_STOP
TF_QRNN_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA
