#include "fo_pool_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

TF_QRNN_NAMESPACE_BEGIN
TF_QRNN_FO_POOL_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto time_major_fo_pool_shape_function = [](InferenceContext* c) {
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

auto batch_major_fo_pool_shape_function = [](InferenceContext* c) {
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
    TF_RETURN_IF_ERROR(
        c->Merge(c->Dim(in_x, 0), c->Dim(in_hinit, 0), &dims[0]));
    TF_RETURN_IF_ERROR(
        c->Merge(c->Dim(in_x, 2), c->Dim(in_hinit, 1), &dims[2]));

    for (int i = 0; i < 3; i++) {
        TF_RETURN_IF_ERROR(
            c->Merge(c->Dim(in_x, i), c->Dim(in_forget, i), &dims[i]));
    }

    TF_RETURN_IF_ERROR(c->Add(c->Dim(in_x, 1),
                              static_cast<tensorflow::int64>(1),
                              &dims[1]));

    c->set_output(0, c->MakeShape(dims));
    return Status::OK();
};

// Register the FoPool operator.
REGISTER_OP("TimeMajorFoPool")
    .Input("x: FT")
    .Input("forget: FT")
    .Input("initial_state: FT")
    .Output("output: FT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Doc(R"doc(QRNN fo_pool operation.)doc")
    .SetShapeFn(time_major_fo_pool_shape_function);

REGISTER_OP("BatchMajorFoPool")
    .Input("x: FT")
    .Input("forget: FT")
    .Input("initial_state: FT")
    .Output("output: FT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Doc(R"doc(QRNN fo_pool operation.)doc")
    .SetShapeFn(batch_major_fo_pool_shape_function);

REGISTER_KERNEL_BUILDER(
    Name("TimeMajorFoPool")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_CPU),
    FoPool<CPUDevice, float, true>);

REGISTER_KERNEL_BUILDER(
    Name("TimeMajorFoPool")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_CPU),
    FoPool<CPUDevice, double, true>);

REGISTER_KERNEL_BUILDER(
    Name("BatchMajorFoPool")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_CPU),
    FoPool<CPUDevice, float, false>);

REGISTER_KERNEL_BUILDER(
    Name("BatchMajorFoPool")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_CPU),
    FoPool<CPUDevice, double, false>);

auto time_major_bwd_fo_pool_shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    ShapeHandle in_h = c->input(0);
    ShapeHandle in_x = c->input(1);
    ShapeHandle in_forget = c->input(2);
    ShapeHandle in_gh = c->input(3);

    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_h, 3, &input),
        "h must have shape [None, None, None] but is " +
        c->DebugString(in_h));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_x, 3, &input),
        "x must have shape [None, None, None] but is " +
        c->DebugString(in_h));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_forget, 3, &input),
        "forget must have shape [None, None, None] but is " +
        c->DebugString(in_forget));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_gh, 3, &input),
        "gh must have shape [None, None, None] but is " +
        c->DebugString(in_gh));

    std::vector<DimensionHandle> dims({
        c->Dim(in_gh, 1),
        c->Dim(in_gh, 2)
    });

    c->set_output(0, in_x);
    c->set_output(1, in_forget);
    c->set_output(2, c->MakeShape(dims));

    return Status::OK();
};

auto batch_major_bwd_fo_pool_shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    ShapeHandle in_h = c->input(0);
    ShapeHandle in_x = c->input(1);
    ShapeHandle in_forget = c->input(2);
    ShapeHandle in_gh = c->input(3);

    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_h, 3, &input),
        "h must have shape [None, None, None] but is " +
        c->DebugString(in_h));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_x, 3, &input),
        "x must have shape [None, None, None] but is " +
        c->DebugString(in_h));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_forget, 3, &input),
        "forget must have shape [None, None, None] but is " +
        c->DebugString(in_forget));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_gh, 3, &input),
        "gh must have shape [None, None, None] but is " +
        c->DebugString(in_gh));

    std::vector<DimensionHandle> dims({
        c->Dim(in_gh, 0),
        c->Dim(in_gh, 2)
    });

    c->set_output(0, in_x);
    c->set_output(1, in_forget);
    c->set_output(2, c->MakeShape(dims));

    return Status::OK();
};

REGISTER_OP("TimeMajorBwdFoPool")
    .Input("h: FT")
    .Input("x: FT")
    .Input("forget: FT")
    .Input("gh: FT")
    .Output("gx: FT")
    .Output("gf: FT")
    .Output("ginitial_state: FT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Doc(R"doc(QRNN fo_pool gradient operation.)doc")
    .SetShapeFn(time_major_bwd_fo_pool_shape_function);

REGISTER_OP("BatchMajorBwdFoPool")
    .Input("h: FT")
    .Input("x: FT")
    .Input("forget: FT")
    .Input("gh: FT")
    .Output("gx: FT")
    .Output("gf: FT")
    .Output("ginitial_state: FT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Doc(R"doc(QRNN fo_pool gradient operation.)doc")
    .SetShapeFn(batch_major_bwd_fo_pool_shape_function);

REGISTER_KERNEL_BUILDER(
    Name("TimeMajorBwdFoPool")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_CPU),
    BwdFoPool<CPUDevice, float, true>);

REGISTER_KERNEL_BUILDER(
    Name("TimeMajorBwdFoPool")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_CPU),
    BwdFoPool<CPUDevice, double, true>);

REGISTER_KERNEL_BUILDER(
    Name("BatchMajorBwdFoPool")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_CPU),
    BwdFoPool<CPUDevice, float, false>);

REGISTER_KERNEL_BUILDER(
    Name("BatchMajorBwdFoPool")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_CPU),
    BwdFoPool<CPUDevice, double, false>);

TF_QRNN_FO_POOL_NAMESPACE_STOP
TF_QRNN_NAMESPACE_STOP
