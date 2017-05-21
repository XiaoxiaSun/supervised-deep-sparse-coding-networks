function net = sparseNet_cifar100_init(varargin)
opts.networkType = 'simplenn' ;
opts.batchSize = 128;
opts = vl_argparse(opts, varargin) ;


% Define network CIFAR10-quick
net.layers = {} ;

n = 1;



K = 4;
Kd = 1;

net = insertSCBlock(net, 3, 3, 16, 1,1);
net = insertSCBlock(net, 3, 16, 16*K, 1,1);

for i = 1:2*n-1
    net = insertSCBlock(net, 3, 16*K, 16*Kd, 1,1);
    net = insertSCBlock(net, 3, 16*Kd, 16*K, 1,1);
end

net = insertSCBlock(net, 3, 16*K, 16*Kd, 1,1);
net = insertSCBlock(net, 3, 16*Kd, 16*K, 1,1);

net = insertSCBlock(net, 3, 16*K, 32*Kd, 2,1);
net = insertSCBlock(net, 3, 32*Kd, 32*K, 1,1);

for i = 1:2*n-2
    net = insertSCBlock(net, 3, 32*K, 32*Kd, 1,1);
    net = insertSCBlock(net, 3, 32*Kd, 32*K, 1,1);
end

net = insertSCBlock(net, 3, 32*K, 32*Kd, 1,1);
net = insertSCBlock(net, 3, 32*Kd, 32*K, 1,1);

net = insertSCBlock(net, 3, 32*K, 64*Kd, 2,1);
net = insertSCBlock(net, 3, 64*Kd, 64*K,  1,1);

for i = 1:2*n-2
    net = insertSCBlock(net, 3, 64*K, 64*Kd, 1,1);
    net = insertSCBlock(net, 3, 64*Kd, 64*K, 1,1);
end

net = insertSCBlock(net, 3, 64*K, 64*Kd, 1,1);
net = insertSCBlock(net, 3, 64*Kd, 64*K, 1,1);
net = insertBnormLastLayer(net, 64*K);

net.layers{end+1} = struct('type', 'pool', ...
    'method', 'avg', ...
    'pool', [8 8], ...
    'stride', 4, ...
    'pad', [0 0 0 0]) ;

N = 64*K;
init_weights = (randn(1, 1, N, 100, 'single'))/(1*N);
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{init_weights, zeros(1,100,'single')}}, ...
    'learningRate', 1e0*[1, 1], 'weightDecay', [1, 1],...
    'stride', 1, ...
    'pad', 0) ;


net.layers{end+1} = struct('type', 'softmaxloss') ;

% Meta parameters
net.meta.inputSize = [32 32 3];
net.meta.trainOpts.learningRate = [1e-1*ones(1,80), 1e-2*ones(1, 80), 1e-3*ones(1, 40)];
net.meta.trainOpts.weightDecay = 5e-4;
net.meta.trainOpts.batchSize = opts.batchSize;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% Fill in default values
net = vl_simplenn_tidy(net) ;
% Switch to DagNN if requested
switch lower(opts.networkType)
    case 'simplenn'
        % done
    case 'dagnn'
        net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
        net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
            {'prediction','label'}, 'error') ;
    otherwise
        assert(false) ;
end




function net = insertSCBlock(net, ksize, ndim_in, ndim_out, stride, lr)

lambda = 0;
lambda2 = 0;
init_weights = (randn(ksize, ksize, ndim_in,  ndim_out, 'single'))/(ksize^2*ndim_in);
net = insertBnormLastLayer(net, ndim_in);

net.layers{end+1} = struct('type', 'sc_layer', ...
    'weights', {{init_weights, lambda*ones(ndim_out, 1, 'single'), lambda2*ones(ndim_out, 1, 'single')}}, ...
    'lambda2', lambda2, 'stride', stride, 'learningRate', lr*[1, 1],  'weightDecay', [1, 1], ...
    'error_rate', 0, 'nonzero_rate', 0, 'active_count', zeros(ndim_out, 1, 'single'));


function net = insertBnormLastLayer(net, ndim)
% --------------------------------------------------------------------
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single'), zeros(ndim, 2, 'single')}}, ...
    'learningRate', [0 0 0.1], 'epsilon', 1e-4, ...
    'weightDecay', [0 0]) ;

