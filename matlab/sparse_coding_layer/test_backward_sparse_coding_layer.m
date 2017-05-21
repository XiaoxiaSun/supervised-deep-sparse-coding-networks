clear; clc; close all;
 
N = 64;
height = 32;
width = 32;
nchannel = 3;
batch_size = 64;
kernel_size = 5;
P = height*width*batch_size;
M = kernel_size^2 * nchannel;
K = 15;

A = gpuArray(normc(randn(M, N, 'single')));
Y = gpuArray(randn(height, width, nchannel, batch_size, 'single'));

param.K = K;
param.height = height;
param.width = width;
param.nchannel = nchannel;
param.kernel_size = kernel_size;
param.batch_size = batch_size;

grad_output = randn(N, P, 'single', 'gpuArray');

tic
 [ grad_A, grad_input ] = backward_sparse_coding_layer( Y, A, grad_output, param );
 toc