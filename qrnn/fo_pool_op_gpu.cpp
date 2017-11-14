#if GOOGLE_CUDA

#include "fo_pool_op_gpu.h"
#include "fo_pool_op_kernel.h"

TF_QRNN_NAMESPACE_BEGIN
TF_QRNN_FO_POOL_NAMESPACE_BEGIN


// Register a GPU kernel for FoPool
// handling permutation ['float']
REGISTER_KERNEL_BUILDER(
    Name("FoPool")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_GPU),
    FoPool<GPUDevice, float>);

TF_QRNN_FO_POOL_NAMESPACE_STOP
TF_QRNN_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA
