#if GOOGLE_CUDA

#include "fo_pool_op_gpu.h"
#include "fo_pool_op_kernel.h"

TF_QRNN_NAMESPACE_BEGIN
TF_QRNN_FO_POOL_NAMESPACE_BEGIN

// Register a GPU kernel for FoPool

/* TIME MAJOR */

REGISTER_KERNEL_BUILDER(
    Name("TimeMajorFoPool")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_GPU),
    FoPool<GPUDevice, float, true>);

REGISTER_KERNEL_BUILDER(
    Name("TimeMajorBwdFoPool")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_GPU),
    BwdFoPool<GPUDevice, float, true>);

REGISTER_KERNEL_BUILDER(
    Name("TimeMajorFoPool")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_GPU),
    FoPool<GPUDevice, double, true>);

REGISTER_KERNEL_BUILDER(
    Name("TimeMajorBwdFoPool")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_GPU),
    BwdFoPool<GPUDevice, double, true>);

/* BATCH MAJOR */

REGISTER_KERNEL_BUILDER(
    Name("BatchMajorFoPool")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_GPU),
    FoPool<GPUDevice, float, false>);

REGISTER_KERNEL_BUILDER(
    Name("BatchMajorBwdFoPool")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_GPU),
    BwdFoPool<GPUDevice, float, false>);

REGISTER_KERNEL_BUILDER(
    Name("BatchMajorFoPool")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_GPU),
    FoPool<GPUDevice, double, false>);

REGISTER_KERNEL_BUILDER(
    Name("BatchMajorBwdFoPool")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_GPU),
    BwdFoPool<GPUDevice, double, false>);


TF_QRNN_FO_POOL_NAMESPACE_STOP
TF_QRNN_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA
