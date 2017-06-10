Supervised Sparse Coding Networks
=============

This code is used for experiments of Supervised Deep Sparse Coding Networks https://arxiv.org/abs/1701.08349 by Xiaoxia Sun, Nasser M. Nasrabadi and Trac D. Tran. 

```
@article{xxsun2017deep,
  author    = {Xiaoxia Sun and Nasser M. Nasrabadi and Trac D. Tran},
  title     = {Supervised Multilayer Sparse Coding Networks for Image Classification},
  journal   = {arXiv preprint arXiv:1701.08349},
  year      = {2017},
}
```



The code is verified on a system of Linux Ubuntu 14.01, CUDA 8.0, with 3 Nvidia Titan X (Pascal) GPUs or 4 Nvidia Tesla P40 GPUs.

With 3 Titan X (Pascal) GPUs, training speed is about 80~90 images/sec on CIFAR-10 using SCN-4 settings in the paper.

With 4 Tesla P40 GPUs, training speed is about 100~120 images/sec on CIFAR-10 using SCN-4 settings in the paper.


The sparse coding layer only has GPU version, GPU is required to run the code

0. **To install the MatConvNet toolbox and the SparseNet**
```matlab
compileSparseNet
```

0. **Before run the experiments, add path*1
```matlab
addpath_sparse_coding_layer
```

0. **To reproduce the result on CIFAR-10**
```matlab
[net_bn, info_bn] = sparseNet_cifar10('expDir', 'data/cifar10-sparseNet', 'gpus', [1, 2, 3, 4], 'batchSize', 128, , 'numSlice', 2);
```

batchSize: minibatch size of stochastic gradient descent.
gpus: indices of gpus to be used. Starting from 1.
numSlice: during backpropagation, a batch of samples are sliced into numSlice to reduce the memory usage. 

0. **To reproduce the result on CIFAR-100:**
```matlab
[net_bn, info_bn] = sparseNet_cifar100('expDir', 'data/cifar100-sparseNet', 'gpus', [1, 2, 3, 4], 'batchSize', 128,  'numSlice', 2);
```

0. **To reproduce the result on STL-10:**
```matlab
[net_bn, info_bn] = sparseNet_stl10('expDir', 'data/stl10-scn', 'gpus', [1,2,3,4], 'batchSize', 16,  'numSlice', 2);
```

0. **To reproduce the result on MNIST:**
```matlab
[net_bn, info_bn] = sparseNet_mnist('expDir', 'data/mnist-scn', 'gpus', [1,2,3,4], 'batchSize', 128, 'numSlice', 2);
```
