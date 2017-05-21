clear; clc; close all;
H = 7; 
W = 7;
nchannel_in = 3;
num_batch = 5;
ksize = 3;
stride =1;
nchannel_out = ksize^2 * nchannel_in;

X = randn(H, W, nchannel_in, num_batch, 'single', 'gpuArray');


%%

% X_col = xx_im2col(X, ksize, stride);
X_col = mexIm2col(X, ksize, stride);
X_rec = mexCol2im(X_col, H, W, nchannel_in, num_batch, ksize, stride);

% start testing mexCol2im
% X_rec = xx_col2im(X_col, ksize, stride, H, W, nchannel_in, num_batch);
% X_rec = mexCol2im(Y, H, W, nchannel_in, num_batch, ksize);

% 
% X_col1 = xx_im2col(X, ksize);
% 
% 
% % start testing mexCol2im
% X_rec1 = xx_col2im(X_col1, ksize, H, W, nchannel_in, num_batch);
