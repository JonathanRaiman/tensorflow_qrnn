import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.ops import gradient_checker
from tensorflow.python.framework import constant_op

from qrnn import fo_pool, _fo_pool

def np_fo_pooling(x, forget, initial_state):
    timesteps, batch, hidden = x.shape
    dst = np.zeros((timesteps + 1, batch, hidden), dtype=x.dtype)
    dst[0] = initial_state
    for ts in range(1, timesteps + 1):
        dst[ts] = (forget[ts - 1]         * x[ts - 1] +
                   (1.0 - forget[ts - 1]) * dst[ts - 1])
    return dst[1:]


class TestFoPool(unittest.TestCase):
    """ Tests the FoPool operator """

    def setUp(self):
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
                return fo_pool(*tf_args)

        # Pin operation to CPU
        cpu_op = _pin_op("/cpu:0", *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)
            cpu_result = S.run(cpu_op)
            self.assertEqual(cpu_result.shape, shape)
            gpu_results = S.run(gpu_ops)
            for gpu_result in gpu_results:
                self.assertEqual(gpu_result.shape, shape)

            slow_result = np_fo_pooling(x, forget, initial_state)
            self.assertTrue(np.allclose(cpu_result, slow_result))
            for gpu_result in gpu_results:
                self.assertTrue(np.allclose(gpu_result, slow_result))

    def test_fo_pool_grad(self):
        """ Test the FoPool Gradient operator """
        # List of type constraint for testing this operator
        type_permutations = [np.float32]

        # Run test with the type combinations above
        for FT in type_permutations:
            self._impl_test_fo_pool_grad(FT)


    def _impl_test_fo_pool_grad(self, FT):
        # Create input variables
        timesteps = 5
        batch_size = 3
        channels = 2
        shape = (timesteps, batch_size, channels)
        output_shape = (timesteps + 1, batch_size, channels)

        x_init = np.random.random(size=shape).astype(FT)
        forget_init = np.random.uniform(0, 1, size=shape).astype(FT)
        initial_state_init = np.random.random(size=(batch_size, channels)).astype(FT)

        with tf.Session() as S:
            x = constant_op.constant(x_init, shape=x_init.shape, dtype=FT, name="x")
            forget = constant_op.constant(forget_init, shape=forget_init.shape, dtype=FT, name="forget")
            initial_state = constant_op.constant(initial_state_init, shape=initial_state_init.shape, dtype=FT, name="initial_state")
            y = _fo_pool(x, forget, initial_state)

            for d in self.gpu_devs + ["cpu"]:
                with tf.device(d):
                    err = gradient_checker.compute_gradient_error(
                      [x, forget, initial_state],
                      [shape, shape, shape[1:]],
                      y, output_shape,
                      x_init_value=[x_init, forget_init, initial_state_init])
                self.assertLess(err, 1e-4)


if __name__ == "__main__":
    unittest.main()
