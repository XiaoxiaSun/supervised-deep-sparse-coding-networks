function [ grad_input, grad_A, grad_lambda  ] = backward_sparse_coding_layer( Y, X, A, lambda1, grad_output, stride, num_slice )
%backward_sparse_coding_layer Perform backpropagataion of sparse coding layer
%   Input:
%       Y: input image, height x width x num_channel x batch_size
%       grad_output: \partial L / \partial x, height x width x N x
%       batch_size
%       X: computed sparse code, height x width x N x batch_size
%        A: dictionary (weights), ksize x ksize x nchannel x N
%   Output:
%       grad_A: gradient for updating dictionary
%       grad_input: gradient of sparse code w.r.t. input feature,
%                           height x width x m



[height, width, ~, batch_size] = size(Y);
[ksize, ~, nchannel, N] = size(A);

M = ksize^2*nchannel;
A = reshape(A, M, N);
grad_output = permute(grad_output, [3, 1, 2, 4]);  %N x height x width x batch_size
tmp_height = floor((height+2*floor(ksize/2) - ksize) / stride) + 1;
tmp_width = floor((width+2*floor(ksize/2) - ksize) / stride) + 1;
Y_col = mexIm2col((Y), ksize, stride);
grad_output = reshape(grad_output, N, tmp_height*tmp_width*batch_size);
X_col = reshape(permute(X, [3, 1, 2, 4]), N, tmp_height*tmp_width*batch_size);
P = size(X_col, 2);
clear X;

% start computing gradient
% param_grad.K = K;
param_grad.height = height;
param_grad.width = width;
param_grad.nchannel = nchannel;
param_grad.kernel_size = ksize;
param_grad.stride = stride;

% num_slice = 3;
size_slice = ceil(batch_size/num_slice);

grad_A = zeros(M, N, 'single', 'gpuArray');
grad_lambda = zeros(N, 1, 'single', 'gpuArray');
grad_input = zeros(height, width, nchannel, batch_size, 'single', 'gpuArray');

for s = 1:num_slice
    
    idx = (s-1)*tmp_height*tmp_width*size_slice+1:min(s*tmp_height*tmp_width*size_slice, P);
    idx_batch = (s-1)*size_slice+1:min(s*size_slice, batch_size);
    if numel(idx_batch)==0
        break;
    else
        param_grad.batch_size = numel(idx_batch);
        
        [tmp_grad_A, tmp_grad_lambda, tmp_grad_input] = get_gradient_batch( X_col(:, idx), Y_col(:, idx), A, lambda1, grad_output(:, idx), param_grad );
        grad_A = grad_A + tmp_grad_A;
        grad_lambda = grad_lambda + tmp_grad_lambda;
        grad_input(:, :, :, idx_batch) = tmp_grad_input;
    end
    
end



grad_A = reshape(grad_A, [ksize, ksize, nchannel, N]);


end
