function [grad_A, grad_lambda, grad_input] = get_gradient_batch( X, Y, A, lambda1, grad_output, param )
% compute gradient for sparse coding layer
% 




[M, N] = size(A);

P = size(X, 2);



% K = param.K;
height = param.height;
width = param.width;
nchannel = param.nchannel;
kernel_size = param.kernel_size;
batch_size = param.batch_size;
stride = param.stride;


epsilon = 0;

act_idx_binary = ((X)>epsilon);  % binary vector, N x P
[linear_act_idx_i, ~] = find((X)>epsilon);
linear_act_idx = find((X)>epsilon);
linear_act_idx = linear_act_idx(:);
linear_act_idx_i = linear_act_idx_i(:);
N_max = max(sum(act_idx_binary, 1));
colapse_idx = sort(act_idx_binary, 1, 'descend');    % N x P
colapse_idx = colapse_idx(1:N_max, :);
colapse_idx = colapse_idx(:);
act_idx_binary = act_idx_binary(:);

act_idx = zeros(N_max, P, 'single', 'gpuArray');
act_idx(colapse_idx) = linear_act_idx_i;
AtA_act = mexGetActiveAtA(A'*A, act_idx) + 1e-6*eye(N_max, N_max, 'single', 'gpuArray');
grad_output = grad_output(:);
grad_output_act = zeros(N_max, P, 'single', 'gpuArray');
grad_output_act(colapse_idx) = grad_output(linear_act_idx);
grad_output_act = reshape(grad_output_act, [N_max, 1, P]);
Beta_act = pagefun(@mldivide, AtA_act, grad_output_act);  % N_max x 1 x P
Beta = zeros(N, P, 'single', 'gpuArray');
Beta(linear_act_idx) = Beta_act(colapse_idx);





grad_lambda = -Beta_act;



clear AtA_act;
grad_lambda_full = zeros(N, P, 'single', 'gpuArray');
grad_lambda_full(act_idx_binary) = grad_lambda(colapse_idx);   % N x P
grad_lambda = sum(grad_lambda_full(:))*ones(N, 1, 'single', 'gpuArray');
clear grad_lambda_full grad_output;






Err = Y - A*X;
X_act = zeros(N_max, P, 'single', 'gpuArray');
X_act(colapse_idx) = X(linear_act_idx);
ABeta = A*Beta;
grad_input = ABeta;

if height*width==kernel_size^2
   grad_input = reshape(grad_input, [1, 1, ]);
else
    grad_input = mexCol2im(grad_input, height, width, nchannel, batch_size, kernel_size, stride);
end


grad_A = mexGetGradA3D(-ABeta, X_act, Err, squeeze(Beta_act));
clear A_act;
grad_A = reshape(grad_A, [M, N_max*P]);

grad_A = reshape(grad_A, [M, N_max*P]);
grad_A = grad_A(:, colapse_idx);
[sort_linear_act_idx_i, sort_grad_A] = sort(linear_act_idx_i, 'ascend');
grad_A = grad_A(:, sort_grad_A);
act_idx_binary_2d = reshape(act_idx_binary, N, P);
maxP = max(sum(act_idx_binary_2d, 2));   % maxP is much smaller than P to save memory
grad_A_full = zeros(M, maxP*N, 'single', 'gpuArray');
act_idx_binary = sort(act_idx_binary_2d, 2, 'descend');    % N x P
act_idx_binary = act_idx_binary(:, 1:maxP)';
act_idx_binary = act_idx_binary(:);
grad_A_full(:, act_idx_binary) = grad_A;
grad_A_full = reshape(grad_A_full, [M, maxP, N]);



grad_A = squeeze(sum(grad_A_full, 2));

clear grad_A_full 





end

%