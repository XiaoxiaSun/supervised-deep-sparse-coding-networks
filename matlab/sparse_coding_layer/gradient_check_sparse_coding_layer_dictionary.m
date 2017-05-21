clear; clc; close all;

gpuDevice(1)

N = 64;
height = 32;
width = 32;
nchannel = 3;
batch_size = 2;
kernel_size = 5;
P = height*width*batch_size;
M = kernel_size^2 * nchannel;
K = 40;
stride = 2;

A = gpuArray(normc(randn(M, N, 'single')));
A = reshape(A, kernel_size, kernel_size, nchannel, N);
% A = randn(kernel_size, kernel_size, nchannel, N, 'single', 'gpuArray');
Y = gpuArray(randn(height, width, nchannel, batch_size, 'single'));

eta = 1e-4;
noise = zeros(kernel_size, kernel_size, nchannel, N, 'single', 'gpuArray');
i = 2; j = 3; c = 2; n = 6;
noise(i, j, c, n) = eta;
% noise = randn(height, width, nchannel, batch_size, 'single', 'gpuArray');

param.K = K;
param.height = height;
param.width = width;
param.nchannel = nchannel;
param.kernel_size = kernel_size;
param.batch_size = batch_size;

height = floor((height+2*floor(kernel_size/2) - kernel_size) / stride) + 1;
width = floor((width+2*floor(kernel_size/2) - kernel_size) / stride) + 1;
grad_output = randn(height, width, N, batch_size, 'single', 'gpuArray');

% gradient checking for grad_input

X_eta = forward_sparse_coding_layer( Y, A+noise, K, stride );
X = forward_sparse_coding_layer( Y, A - noise, K, stride );
grad_input_empirical = sum(grad_output(:) .* (X_eta(:) - X(:))/(2*eta));


%%
X_clean = forward_sparse_coding_layer( Y, A, K, stride );
 [ grad_input, grad_A ] = backward_sparse_coding_layer( Y, X_clean, A, grad_output, K, stride );
 grad_input_computed = (grad_A(i, j, c, n));
 
fprintf(...
  'der: empirical: %f, computed: %f, error: %.2f %%\n', ...
  grad_input_empirical, grad_input_computed, ...
  abs(1 - grad_input_empirical/grad_input_computed)*100) ;

%%
