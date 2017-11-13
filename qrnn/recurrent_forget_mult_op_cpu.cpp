#include "recurrent_forget_mult_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

TF_QRNN_NAMESPACE_BEGIN
TF_QRNN_RECURRENT_FORGET_MULT_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    // TODO. Check shape and dimension sizes for 'x'
    ShapeHandle in_x = c->input(0);
    // Assert 'x' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_x, 3, &input),
        "x must have shape [None, None, None] but is " +
        c->DebugString(in_x));

    // TODO. Check shape and dimension sizes for 'forget'
    ShapeHandle in_forget = c->input(1);
    // Assert 'forget' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_forget, 3, &input),
        "forget must have shape [None, None, None] but is " +
        c->DebugString(in_forget));

    // TODO: Supply a proper shapes for output variables here,
    // usually derived from input shapes
    ShapeHandle output_shape = c->MakeShape({
         c->Dim(in_x, 0),
         c->Dim(in_x, 1),
         c->Dim(in_x, 2)});

    c->set_output(0, output_shape);
    // printf("output shape %s\\n", c->DebugString(output_shape).c_str());;
    return Status::OK();
};

// Register the RecurrentForgetMult operator.
REGISTER_OP("RecurrentForgetMult")
    .Input("x: FT")
    .Input("forget: FT")
    .Output("output: FT")
    .Attr("FT: {float} = DT_FLOAT")
    .Doc(R"doc(QRNN nonlinearity.)doc")
    .SetShapeFn(shape_function);


// Register a CPU kernel for RecurrentForgetMult
// handling permutation ['float']
REGISTER_KERNEL_BUILDER(
    Name("RecurrentForgetMult")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_CPU),
    RecurrentForgetMult<CPUDevice, float>);

TF_QRNN_RECURRENT_FORGET_MULT_NAMESPACE_STOP
TF_QRNN_NAMESPACE_STOP
