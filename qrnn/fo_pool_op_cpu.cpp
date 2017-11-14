#include "fo_pool_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

TF_QRNN_NAMESPACE_BEGIN
TF_QRNN_FO_POOL_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto fo_pool_shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    ShapeHandle in_x = c->input(0);
    // Assert 'x' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_x, 3, &input),
        "x must have shape [None, None, None] but is " +
        c->DebugString(in_x));
    ShapeHandle in_forget = c->input(1);
    // Assert 'forget' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_forget, 3, &input),
        "forget must have shape [None, None, None] but is " +
        c->DebugString(in_forget));

    ShapeHandle in_hinit = c->input(2);
    // Assert 'hinit' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_hinit, 2, &input),
        "hinit must have shape [None, None] but is " +
        c->DebugString(in_hinit));

    std::vector<DimensionHandle> dims(3);
    for (int i = 1; i < 3; i++) {
        TF_RETURN_IF_ERROR(
            c->Merge(c->Dim(in_x, i), c->Dim(in_hinit, i - 1), &dims[i]));
    }


    for (int i = 0; i < 3; i++) {
        TF_RETURN_IF_ERROR(
            c->Merge(c->Dim(in_x, i), c->Dim(in_forget, i), &dims[i]));
    }

    TF_RETURN_IF_ERROR(c->Add(c->Dim(in_x, 0),
                              static_cast<tensorflow::int64>(1),
                              &dims[0]));

    c->set_output(0, c->MakeShape(dims));
    return Status::OK();
};

// Register the FoPool operator.
REGISTER_OP("FoPool")
    .Input("x: FT")
    .Input("forget: FT")
    .Input("hinit: FT")
    .Output("output: FT")
    .Attr("FT: {float} = DT_FLOAT")
    .Doc(R"doc(QRNN fo_pool operation.)doc")
    .SetShapeFn(fo_pool_shape_function);


auto bwd_fo_pool_shape_function = [](InferenceContext* c) {
    return Status::OK();
};

REGISTER_OP("BwdFoPool")
    .Input("h: FT")
    .Input("x: FT")
    .Input("f: FT")
    .Input("gh: FT")
    .Output("gf: FT")
    .Output("gx: FT")
    .Output("ginitial_state: FT")
    .Attr("FT: {float} = DT_FLOAT")
    .Doc(R"doc(QRNN fo_pool gradient operation.)doc")
    .SetShapeFn(bwd_fo_pool_shape_function);


// Register a CPU kernel for FoPool
// handling permutation ['float']
REGISTER_KERNEL_BUILDER(
    Name("FoPool")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_CPU),
    FoPool<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("BwdFoPool")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_CPU),
    FoPool<CPUDevice, float>);

// Status FoPoolGrad(const Scope& scope, const Operation& op,
//                   const std::vector<Output>& grad_inputs,
//                   std::vector<Output>* grad_outputs) {
//   grad_outputs->reserve(3);
//   auto grad_op = BwdFoPool(scope,
//                            /* h = */op.output(0),
//                            /* x = */op.input(0)
//                            /* f = */ op.input(1),
//                            /* gh = */grad_inputs[0]);
//   for (const Output& o : grad_op.output) {
//     grad_outputs->emplace_back(o);
//   }
//   return scope.status();
// }

TF_QRNN_FO_POOL_NAMESPACE_STOP
TF_QRNN_NAMESPACE_STOP
