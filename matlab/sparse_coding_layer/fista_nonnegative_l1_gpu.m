function x_old = fista_nonnegative_l1_gpu(D, Y,  lambda)

DtD = D'*D;
DtY = D'*Y;

maxIter = 50;
L = max(eig(DtD));
Linv = 1/L;
lambdaLinv = lambda*Linv;
[M, N] = size(D);

P = size(Y, 2);
x_old = zeros(N, P, 'double', 'gpuArray');
y_old = x_old;
t_old = 1;

%% MAIN LOOP
A = eye(N, 'double', 'gpuArray') - Linv*(DtD);
const_x = Linv*DtY - lambdaLinv;

for iter = 1:maxIter
    x_new = A*y_old + const_x;
    x_new = max(x_new, 0);
    t_new = 0.5*(1 + sqrt(1 + 4*t_old^2));
    y_new = (1 + (t_old - 1)/t_new) * x_new -  (t_old - 1)/t_new *x_old;
    %% update
    x_old = x_new;
    t_old = t_new;
    y_old = y_new;
end

end


