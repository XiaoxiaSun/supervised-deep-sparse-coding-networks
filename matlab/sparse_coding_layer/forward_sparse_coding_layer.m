function X = forward_sparse_coding_layer( Y, A, lambda, stride, testMode )
%forward_sparse_coding_layer Perform forward pass of sparse coding layer
%   input:
%       A: dictionary (weights), ksize x ksize x nchannel x N
%       Y: input image, height x width x num_channel x batch_size
%   output:

% all data must be single format
% if ~isa(Y, 'single')
%     error('Input image must be single');
% end
%
% if ~isa(A, 'single')
%     error('Input dictionary must be single');
% end

% kernel_size = param.kernel_size;
% K = param.K;
% error_rate = 0;
% nonzero_rate = 0;

[height, width, ~, batch_size] = size(Y);
[ksize, ~, nchannel, N] = size(A);

A = reshape(A, ksize^2*nchannel, N);

% if M~=nchannel*kernel_size^2
%     error('kernel size and dictionary feature number mismatch!');
% end

% convert image to column, M x height*width*batch_size (in order)
% Y_col = xx_im2col(Y, ksize, stride);

% if height*width==ksize^2
%     Y_col = reshape(Y, height*width*nchannel, batch_size);
% elseif ksize==1
%     Y = permute(Y, [3, 1, 2, 4]);
%     Y_col = reshape(Y, nchannel, height*width*batch_size);
% else
Y_col = mexIm2col((Y), ksize, stride);
% end
% Y_col = double(Y_col);
% X_col is sparse code
% X_col = batch_omp( Y_col, A, K);   % N x height*width*batch_size
% X_col = batch_omp( Y_col, A, 15);   % N x height*width*batch_size
% X_col = batch_omp_group_accurate( Y_col, A, K, group_size);

% Y_col = Y_col ./ max(sum(Y_col.^2).^(1/2)+1e-4);
% if(testMode)
    X_col = fista_nonnegative_l1_gpu(A, Y_col,  lambda);
% else
%     X_col = fista_nonnegative_l1_dropout_gpu(A, Y_col,  lambda);
% end
% X_col = ista_nonnegative_l1_gpu(A, Y_col,  lambda);
% [X_col, error_rate, nonzero_rate] = admm_lasso_gpu(A, Y_col,  lambda, 0);

%     size(Y)
%     size(A)

% if height*width==ksize^2
%     X = reshape(X_col, [1, 1, N, batch_size]);
% %     size(Y)
% %     size(A)
% elseif ksize==1
%     X = reshape(X_col, [N, height, width, batch_size]);
%     X = permute(X, [2, 3, 1, 4]);  % height x width x N x batch_size
% else
% convert sparse codes back to image
h_out = floor((height+2*floor(ksize/2) - ksize) / stride) + 1;
w_out = floor((width+2*floor(ksize/2) - ksize) / stride) + 1;
X = reshape(X_col, [N, h_out, w_out, batch_size]);
X = permute(X, [2, 3, 1, 4]);  % height x width x N x batch_size
% end


end

