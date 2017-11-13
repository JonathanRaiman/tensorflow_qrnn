#ifndef QRNN_RECURRENT_FORGET_MULT_OP_H
#define QRNN_RECURRENT_FORGET_MULT_OP_H

// tf_qrnn namespace start and stop defines
#define TF_QRNN_NAMESPACE_BEGIN namespace tf_qrnn {
#define TF_QRNN_NAMESPACE_STOP }

//  namespace start and stop defines
#define TF_QRNN_RECURRENT_FORGET_MULT_NAMESPACE_BEGIN namespace  {
#define TF_QRNN_RECURRENT_FORGET_MULT_NAMESPACE_STOP }

TF_QRNN_NAMESPACE_BEGIN
TF_QRNN_RECURRENT_FORGET_MULT_NAMESPACE_BEGIN

// General definition of the RecurrentForgetMult op, which will be specialised in:
//   - recurrent_forget_mult_op_cpu.h for CPUs
//   - recurrent_forget_mult_op_gpu.cuh for CUDA devices
// Concrete template instantions of this class are provided in:
//   - recurrent_forget_mult_op_cpu.cpp for CPUs
//   - recurrent_forget_mult_op_gpu.cu for CUDA devices
template <typename Device, typename FT>
class RecurrentForgetMult {};

TF_QRNN_RECURRENT_FORGET_MULT_NAMESPACE_STOP
TF_QRNN_NAMESPACE_STOP

#endif // #ifndef QRNN_RECURRENT_FORGET_MULT_OP_H