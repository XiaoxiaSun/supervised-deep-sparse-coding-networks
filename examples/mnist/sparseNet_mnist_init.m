function net = sparseNet_mnist_init(varargin)
opts.networkType = 'simplenn' ;
opts.batchSize = 128;
opts = vl_argparse(opts, varargin) ;


net.layers = {} ;

n = 1;

blockfn = @insertSCBlock ;

K = 4;
Kd = 1;

net = blockfn(net, 3, 1, 8, 1,1);
net = blockfn(net, 3, 8, 16*K, 1,1);



for i = 1:2*n-1
    net = blockfn(net, 3, 16*K, 16*Kd, 1,1);
    net = blockfn(net, 3, 16*Kd, 16*K, 1,1);
end

net = blockfn(net, 3, 16*K, 16*Kd, 1,1);
net = blockfn(net, 3, 16*Kd, 16*K, 1,1);


net = blockfn(net, 3, 16*K, 32*Kd, 2,1);
net = blockfn(net, 3, 32*Kd, 32*K, 1,1);
for i = 1:2*n-2
    net = blockfn(net, 3, 32*K, 32*Kd, 1,1);
    net = blockfn(net, 3, 32*Kd, 32*K, 1,1);
end
net = blockfn(net, 3, 32*K, 32*Kd, 1,1);
net = blockfn(net, 3, 32*Kd, 32*K, 1,1);

net = blockfn(net, 3, 32*K, 64*Kd, 2,1);
net = blockfn(net, 3, 64*Kd, 64*K,  1,1);
for i = 1:2*n-2
    net = blockfn(net, 3, 64*K, 64*Kd, 1,1);
    net = blockfn(net, 3, 64*Kd, 64*K, 1,1);
end

net = blockfn(net, 3, 64*K, 64*Kd, 1,1);
net = blockfn(net, 3, 64*Kd, 64*K, 1,1);

net = insertBnormLastLayer(net, 64*K);

%
net.layers{end+1} = struct('type', 'pool', ...
    'method', 'avg', ...
    'pool', [7 7], ...
    'stride', 4, ...
    'pad', [0 0 0 0]) ; % Emulate caffe




N = 64*K;
init_weights = (randn(1, 1, N, 10, 'single'))/(1*N);
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{init_weights, zeros(1,10,'single')}}, ...
    'learningRate', 1e0*[1, 1], 'weightDecay', [1, 1],...
    'stride', 1, ...
    'pad', 0) ;





net.layers{end+1} = struct('type', 'softmaxloss') ;




% Meta parameters
net.meta.inputSize = [28 28 1];
net.meta.trainOpts.learningRate = [1e-1*ones(1,10), 1e-2*ones(1, 10), 1e-3*ones(1, 5)];
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

lambda = 0*1e-3;
lambda2 = 0*1e-6;
%
init_weights = (randn(ksize, ksize, ndim_in,  ndim_out, 'single'))/(ksize^2*ndim_in);

net.layers{end+1} = struct('type', 'sc_layer', ...
    'weights', {{init_weights, lambda*ones(ndim_out, 1, 'single'), lambda2*ones(ndim_out, 1, 'single')}}, ...
    'lambda2', lambda2, 'stride', stride, 'learningRate', lr*[1, 1],  'weightDecay', [1, 1], ...
    'error_rate', 0, 'nonzero_rate', 0);

net = insertBnormLastLayer(net, ndim_out);


function net = insertBnormLastLayer(net, ndim)
% --------------------------------------------------------------------
% assert(isfield(net.layers{l}, 'weights'));
% ndim = size(net.layers{l}.weights{1}, 4);
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single'), zeros(ndim, 2, 'single')}}, ...
    'learningRate', [0 0 0.1], 'epsilon', 1e-4, ...
    'weightDecay', [0 0]) ;
% net.layers{l}.biases = [] ;
% net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;

