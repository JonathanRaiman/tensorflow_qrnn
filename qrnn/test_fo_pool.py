import unittest

import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib
from tensorflow.python.ops import gradient_checker
from tensorflow.python.framework import constant_op

from qrnn import fo_pool, time_major_fo_pool_unsliced, batch_major_fo_pool_unsliced

def np_fo_pooling(x, forget, initial_state, time_major):
    if not time_major:
        return np.transpose(np_fo_pooling(np.transpose(x, (1, 0, 2)),
                                          np.transpose(forget, (1, 0, 2)),
                                          initial_state,
                                          time_major=True), (1, 0, 2))
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
        type_permutations = [np.float32, np.float64]

        # Run test with the type combinations above
        for FT in type_permutations:
            for time_major in [False]:
                self._impl_test_fo_pool(FT, time_major)

    def _impl_test_fo_pool(self, FT, time_major):
        """ Implementation of the FoPool operator test """
        # Create input variables
        timesteps = 20
        batch_size = 32
        channels = 64
        if time_major:
            shape = (timesteps, batch_size, channels)
        else:
            shape = (batch_size, timesteps, channels)
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
                return fo_pool(*tf_args, time_major=time_major)

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
            expected = np_fo_pooling(x, forget, initial_state, time_major=time_major)
            self.assertTrue(np.allclose(cpu_result, expected))
            for gpu_result in gpu_results:
                self.assertTrue(np.allclose(gpu_result, expected))

    def test_time_major_fo_pool_grad(self):
        """ Test the FoPool Gradient operator """
        # List of type constraint for testing this operator
        type_permutations = [(np.float32, 1e-2), (np.float64, 1e-4)]

        # Run test with the type combinations above
        for FT, tolerance in type_permutations:
            self._impl_test_time_major_fo_pool_grad(FT, tolerance)


    def _impl_test_time_major_fo_pool_grad(self, FT, tolerance):
        shape = (5, 3, 2)
        np_args = [np.random.random(size=shape).astype(FT),
                   np.random.uniform(0, 1, size=shape).astype(FT),
                   np.random.random(size=shape[1:]).astype(FT)]
        with tf.Session() as S:
            tf_args = [constant_op.constant(arg, shape=arg.shape, dtype=FT) for arg in np_args]
            y = tf.reduce_sum(time_major_fo_pool_unsliced(*tf_args))
            for d in ["cpu"] + self.gpu_devs:
                with tf.device(d):
                    err = gradient_checker.compute_gradient_error(
                      tf_args, [arg.shape for arg in np_args], y, [],
                      x_init_value=np_args)
                self.assertLess(err, tolerance)

    def test_batch_major_fo_pool_grad(self):
        """ Test the FoPool Gradient operator """
        # List of type constraint for testing this operator
        type_permutations = [(np.float32, 1e-2), (np.float64, 1e-4)]

        # Run test with the type combinations above
        for FT, tolerance in type_permutations:
            self._impl_test_batch_major_fo_pool_grad(FT, tolerance)


    def _impl_test_batch_major_fo_pool_grad(self, FT, tolerance):
        shape = (3, 5, 2)
        np_args = [np.random.random(size=shape).astype(FT),
                   np.random.uniform(0, 1, size=shape).astype(FT),
                   np.random.random(size=(shape[0], shape[-1])).astype(FT)]
        with tf.Session() as S:
            tf_args = [constant_op.constant(arg, shape=arg.shape, dtype=FT) for arg in np_args]
            y = tf.reduce_sum(batch_major_fo_pool_unsliced(*tf_args))
            for d in ["cpu"] + self.gpu_devs:
                with tf.device(d):
                    err = gradient_checker.compute_gradient_error(
                      tf_args, [arg.shape for arg in np_args], y, [],
                      x_init_value=np_args)
                self.assertLess(err, tolerance)


if __name__ == "__main__":
    unittest.main()
