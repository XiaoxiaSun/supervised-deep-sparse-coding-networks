clear; clc; close all;

M = 4; 
N = 10;
P = 3;

A = randn(M, 1, P, 'single', 'gpuArray');
X = randn(1, N, P, 'single', 'gpuArray');
E = randn(M, 1, P, 'single', 'gpuArray');
B = randn(1, N, P, 'single', 'gpuArray');

grad = pagefun(@mtimes, A, X) + pagefun(@mtimes, E, B);


A = squeeze(A);
X = squeeze(X);
E = squeeze(E);
B = squeeze(B);
temp = mexGetGradA3D(A, X, E, B);

norm(grad(:) - temp(:))