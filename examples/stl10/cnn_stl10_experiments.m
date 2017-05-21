%% Experiment with the cnn_cifar10_fc_bnorm
[net_bn, info_bn] = sparseNet_stl10('expDir', 'data/stl10-scn', 'gpus', [1,2,3], 'batchSize', 16);

