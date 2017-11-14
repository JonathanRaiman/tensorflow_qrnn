import tensorflow as tf
from os.path import join, dirname, realpath

SCRIPT_DIR = dirname(realpath(__file__))

qrnn_lib = tf.load_op_library(join(SCRIPT_DIR, "qrnn_lib.so"))

_fo_pool = qrnn_lib.fo_pool
bwd_fo_pool = qrnn_lib.bwd_fo_pool

@tf.RegisterGradient("FoPool")
def _fo_pool_grad(op, grad):
    return bwd_fo_pool(h=op.outputs[0], x=op.inputs[0],
                       forget=op.inputs[1], gh=grad)


def fo_pool(x, forget, initial_state=None):
    if initial_state is None:
        initial_state = tf.zeros((tf.shape(x)[1], tf.shape(x)[2]), dtype=tf.dtype)
    return _fo_pool(x, forget, initial_state)[1:]
