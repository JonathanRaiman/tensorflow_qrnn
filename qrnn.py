"""
Quasi-Recurrent Neural Network (QRNN) for Tensorflow
----------------------------------------------------

This repository contains a Tensorflow implementation of
[Salesforce Research](https://einstein.ai/)'s
[Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576)
paper. It supports batch-major or time-major inputs in
single or double precision.

From the authors:
> The QRNN provides similar accuracy to the LSTM but can be betwen
> 2 and 17 times faster than the highly optimized NVIDIA cuDNN
> LSTM implementation depending on the use case.

If you use this code or their results in your research, you should cite:

@article{bradbury2016quasi,
  title={{Quasi-Recurrent Neural Networks}},
  author={Bradbury, James and Merity, Stephen and Xiong, Caiming and Socher, Richard},
  journal={International Conference on Learning Representations (ICLR 2017)},
  year={2017}
}

Usage
-----

Use QRNNs as you would use LSTMs or RNNs, to
encode order-specific information:

```
import qrnn

# input sequence in Batch, Time, Channels format:
inputs = tf.placeholder(tf.float32, [None, None, 128])
encoded = qrnn.qrnn(inputs)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  out = sess.run(encoded, {inputs: my_data})
```

"""
import tensorflow as tf
from os.path import join, dirname, realpath

SCRIPT_DIR = dirname(realpath(__file__))

def get_ext_filename(ext_name):
    from distutils.sysconfig import get_config_var
    ext_path = ext_name.split('.')
    ext_suffix = get_config_var('EXT_SUFFIX')
    return join(*ext_path) + ext_suffix


qrnn_lib = tf.load_op_library(join(SCRIPT_DIR, get_ext_filename("qrnn_lib")))

time_major_fo_pool_unsliced = qrnn_lib.time_major_fo_pool
time_major_bwd_fo_pool = qrnn_lib.time_major_bwd_fo_pool

batch_major_fo_pool_unsliced = qrnn_lib.batch_major_fo_pool
batch_major_bwd_fo_pool = qrnn_lib.batch_major_bwd_fo_pool

@tf.RegisterGradient("TimeMajorFoPool")
def _fo_pool_grad(op, grad):
    return time_major_bwd_fo_pool(h=op.outputs[0], x=op.inputs[0],
                                  forget=op.inputs[1], gh=grad)

@tf.RegisterGradient("BatchMajorFoPool")
def _fo_pool_grad(op, grad):
    return batch_major_bwd_fo_pool(h=op.outputs[0], x=op.inputs[0],
                                   forget=op.inputs[1], gh=grad)


def fo_pool(x, forget, initial_state=None, time_major=False):
    """Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.
    Args:
        x: Tensor, input values in [Batch, Time, Channels] format,
           float32 or double
           or [Time, Batch, Channels] if time_major
        forget: Tensor, input values in [Batch, Time, Channels] format,
           float32 or double. Usually in the range 0-1.
           or [Time, Batch, Channels] if time_major
        initial_state: Tensor, initial hidden state values in [Batch, Channels] format,
           float32 or double.

    Returns:
        Tensor: fo_pooled output, [Batch, Time, Channels] format
                or [Time, Batch, Channels] if time_major
    """
    if initial_state is None:
        initial_state = tf.zeros((tf.shape(x)[1] if time_major else tf.shape(x)[0],
                                  tf.shape(x)[2]), dtype=tf.dtype)
    if time_major:
        return time_major_fo_pool_unsliced(x, forget, initial_state)[1:]
    else:
        return batch_major_fo_pool_unsliced(x, forget, initial_state)[:, 1:]


def qrnn(inputs, num_outputs, window=2, output_gate=True,
         activation_fn=tf.tanh, gate_activation_fn=tf.nn.sigmoid,
         padding="SAME", initial_state=None, time_major=False, scope=None,
         **kwargs):
    """Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.
    Args:
        inputs: Tensor, input values in [Batch, Time, Channels] format,
                float32 or double, or [Time, Batch, Channels] if time_major
        window: int, number of values each gating depends on (default=2).
        num_outputs: int, Number of output channels
        keep_prob: float, zoneout dropout probability
        is_training: bool, whether to apply dropout mask
        output_gate: bool, use a gating mechanism on the output
        activation_fn: function, default tanh
        gate_activation_fn: function, default sigmoid
        padding: str, SAME or VALID.
        initial_state: Tensor/None, optional, initializes the QRNN to that value.
        time_major: bool, whether inputs have time-dimension first or second.
        scope: str/None, what to prefix the name the variables under this layer.

    Returns:
        Tensor : qrnn_output, [Batch, Time, Channels] or
                [Time, Batch, Channels] if time_major
    """
    with tf.variable_scope(scope or "QRNNLayer"):
        conv1d_channels = 3 * num_outputs if output_gate else 2 * num_outputs
        if time_major:
            # go to batch_major for convolution if needed
            inputs_batch_major = tf.transpose(inputs, (1, 0, 2), name="InputsBatchMajor")
        else:
            inputs_batch_major = inputs
        gate_values = tf.layers.conv1d(inputs_batch_major,
                                       filters=conv1d_channels,
                                       kernel_size=window,
                                       strides=1,
                                       data_format="channels_last",
                                       name="QRNNConv1D",
                                       padding=padding,
                                       **kwargs)
        if time_major:
            # return to time_major if needed
            gate_values = tf.transpose(gate_values, (1, 0, 2),
                                       name="GateValuesTimeMajor")

        gate_values = tf.split(gate_values, 3 if output_gate else 2, axis=2)
        if output_gate:
            x, forget, output = gate_values
        else:
            x, forget = gate_values

        with tf.name_scope("GateActivations"):
            if activation_fn is not None:
                x = activation_fn(x)
            if gate_activation_fn is not None:
                forget = gate_activation_fn(forget)

        with tf.name_scope("FoPool"):
            c = fo_pool(x, forget, initial_state=initial_state,
                        time_major=time_major)

        with tf.name_scope("OutputGate"):
            h = gate_activation_fn(c) if output_gate else c
    return h, c
