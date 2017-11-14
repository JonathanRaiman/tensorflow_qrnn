import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from os.path import join, dirname, realpath

SCRIPT_DIR = dirname(realpath(__file__))

def np_fo_pooling(x, forget, initial_state):
    timesteps, batch, hidden = x.shape
    dst = np.zeros((timesteps + 1, batch, hidden), dtype=x.dtype)
    dst[0] = initial_state
    for ts in range(1, timesteps + 1):
        dst[ts] = (forget[ts - 1]         * x[ts - 1] +
                   (1.0 - forget[ts - 1]) * dst[ts - 1])
    return dst

class TestFoPool(unittest.TestCase):
    """ Tests the FoPool operator """

    def setUp(self):
        # Load the custom operation library
        self.qrnn = tf.load_op_library(join(SCRIPT_DIR, "qrnn.so"))
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                                if d.device_type == 'GPU']

    def test_fo_pool(self):
        """ Test the FoPool operator """
        # List of type constraint for testing this operator
        type_permutations = [np.float32]

        # Run test with the type combinations above
        for FT in type_permutations:
            self._impl_test_fo_pool(FT)

    def _impl_test_fo_pool(self, FT):
        """ Implementation of the FoPool operator test """
        # Create input variables
        timesteps = 20
        batch_size = 32
        channels = 64
        shape = (timesteps, batch_size, channels)
        output_shape = (timesteps + 1, batch_size, channels)
        x = np.random.random(size=shape).astype(FT)
        forget = np.random.uniform(0, 1, size=shape).astype(FT)
        initial_state = np.random.random(size=(batch_size, channels)).astype(FT)

        # Argument list
        np_args = [x, forget, initial_state]
        # Argument string name list
        arg_names = ["x", "forget", "initial_state"]
        # Constructor tensorflow variables
        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.qrnn.fo_pool(*tf_args)

        # Pin operation to CPU
        cpu_op = _pin_op("/cpu:0", *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)
            cpu_result = S.run(cpu_op)
            self.assertEqual(cpu_result.shape, output_shape)
            gpu_results = S.run(gpu_ops)
            for gpu_result in gpu_results:
                self.assertEqual(gpu_result.shape, output_shape)

            slow_result = np_fo_pooling(x, forget, initial_state)
            self.assertTrue(np.allclose(cpu_result, slow_result))
            for gpu_result in gpu_results:
                self.assertTrue(np.allclose(gpu_result, slow_result))


if __name__ == "__main__":
    unittest.main()
