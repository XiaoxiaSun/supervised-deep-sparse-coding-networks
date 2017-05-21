clear; clc; close all;
H = 32; 
W = 32;
stride = 2;
nchannel_in = 3;
num_batch = 1;
ksize = 1;
pad_size = floor(ksize/2);
nchannel_out = ksize^2 * nchannel_in;

X = randn(H, W, nchannel_in, num_batch, 'single', 'gpuArray');
Xt = permute(X, [2, 1, 3, 4]);

%%
Y = mexIm2col(Xt, ksize, stride);
H_out = floor((H+2*pad_size)/stride);
W_out = floor((W+2*pad_size)/stride);
% Y = reshape(Y, nchannel_out, H_out*W_out*num_batch);


% Y_cpu = im2col_cpu( X, ksize );


