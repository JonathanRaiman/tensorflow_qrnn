import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from os.path import join, dirname, realpath

SCRIPT_DIR = dirname(realpath(__file__))

class TestRecurrentForgetMult(unittest.TestCase):
    """ Tests the RecurrentForgetMult operator """

    def setUp(self):
        # Load the custom operation library
        self.qrnn = tf.load_op_library(join(SCRIPT_DIR, "qrnn.so"))
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                                if d.device_type == 'GPU']

    def test_recurrent_forget_mult(self):
        """ Test the RecurrentForgetMult operator """
        # List of type constraint for testing this operator
        type_permutations = [np.float32]

        # Run test with the type combinations above
        for FT in type_permutations:
            self._impl_test_recurrent_forget_mult(FT)

    def _impl_test_recurrent_forget_mult(self, FT):
        """ Implementation of the RecurrentForgetMult operator test """
        # Create input variables
        x = np.random.random(size=[1, 1, 1]).astype(FT)
        forget = np.random.random(size=[1, 1, 1]).astype(FT)


        # Argument list
        np_args = [x, forget]
        # Argument string name list
        arg_names = ['x', 'forget']
        # Constructor tensorflow variables
        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.qrnn.recurrent_forget_mult(*tf_args)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)
            S.run(cpu_op)
            S.run(gpu_ops)

if __name__ == "__main__":
    unittest.main()
