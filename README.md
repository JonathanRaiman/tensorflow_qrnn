# Quasi-Recurrent Neural Network (QRNN) for Tensorflow

(Work in progress)

This repository contains a Tensorflow implementation of [Salesforce Research](https://einstein.ai/)'s [Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576) paper.

The QRNN provides similar accuracy to the LSTM but can be betwen 2 and 17 times faster than the highly optimized NVIDIA cuDNN LSTM implementation depending on the use case.

To install, simply run:

`pip install tensorflow_qrnn`

If you use this code or their results in your research, you should cite:

```
@article{bradbury2016quasi,
  title={{Quasi-Recurrent Neural Networks}},
  author={Bradbury, James and Merity, Stephen and Xiong, Caiming and Socher, Richard},
  journal={International Conference on Learning Representations (ICLR 2017)},
  year={2017}
}
```

The original PyTorch implementation of the QRNN can be found [here](https://github.com/salesforce/pytorch-qrnn).

### Requirements

- Tensorflow 1.4 (`pip install tensorflow` or `pip install tensorflow-gpu`)
- GCC
- CUDA (optional, needed for GPU support)

### Testing

```
python qrnn/test_recurrent_forget_mult.py
```

### TODOs

- Gradient tests
- PyPi registration + setup.py file
