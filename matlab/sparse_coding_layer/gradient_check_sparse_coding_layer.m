% clear; clc; close all;

% gpuDevice(1)
T = 100;
err = zeros(T, 1);
ave = 0;
for t = 1:T

N = 512;
height = 14;
width = 14;
nchannel = 64;
batch_size = 3;
kernel_size =5;
P = height*width*batch_size;
M = kernel_size^2 * nchannel;
lambda = 0.01;
stride = 2;

A = gpuArray((randn(M, N, 'single')));
% A(A<0) = 0;
A = reshape(A, kernel_size, kernel_size, nchannel, N);
% A = randn(kernel_size, kernel_size, nchannel, N, 'single', 'gpuArray');
Y = (randn(nchannel, height*width*batch_size, 'single', 'gpuArray'));
% Y = abs(Y);
Y = reshape(Y, [nchannel, height, width, batch_size]);
Y = permute(Y, [2, 3, 1, 4]);
% Y = gpuArray(randn(height, width, nchannel, batch_size, 'single'));
eta = 1e-5;
noise = zeros(height, width, nchannel, batch_size, 'single', 'gpuArray');
i = randi(height); j = randi(width); c = randi(nchannel); b = randi(batch_size);
noise(i, j, c, b) = eta;

% noise = randn(height, width, nchannel, batch_size, 'single', 'gpuArray');

% param.lambda = lambda;
% param.height = height;
% param.width = width;
% param.nchannel = nchannel;
% param.kernel_size = kernel_size;
% param.batch_size = batch_size;
%%

height = floor((height+2*floor(kernel_size/2) - kernel_size) / stride) + 1;
width = floor((width+2*floor(kernel_size/2) - kernel_size) / stride) + 1;
grad_output = randn(height, width, N, batch_size, 'single', 'gpuArray');

%%
% gradient checking for grad_input



%%
X_eta = forward_sparse_coding_layer( Y + noise, A, lambda,  stride );
X = forward_sparse_coding_layer( Y - noise, A, lambda, stride );

% X_eta = single(X_eta);
% X = single(X);

grad_input_empirical = sum(grad_output(:) .* (X_eta(:) - X(:))/(2*eta));


%%
X_clean = forward_sparse_coding_layer_single( Y, A, lambda, stride );
 [ grad_input, grad_A ] = backward_sparse_coding_layer( Y, X_clean, A, lambda, grad_output, stride );
 grad_input_computed = (grad_input(i, j, c, b));
 err(t) = gather(abs(1 - grad_input_empirical/grad_input_computed)*100);
fprintf(...
  'der: empirical: %f, computed: %f, error: %.2f %%\n', ...
  grad_input_empirical, grad_input_computed, ...
  abs(1 - grad_input_empirical/grad_input_computed)*100) ;
end

mean(err)

%%
