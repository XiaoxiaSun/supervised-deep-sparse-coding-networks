function [net, info] = sparseNet_stl10(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.modelType = 'sparseNet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(vl_rootnn, 'data', ...
  sprintf('stl10-%s', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.gpus = [1];
opts.batchSize = 128;  % default batch size

opts.numSlice = 3;


opts.dataDir = fullfile(vl_rootnn, 'data','stl10') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = false ;
opts.contrastNormalization = false ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;



opts.border = [4 4 4 4]*3; % tblr
if ~isfield(opts.train, 'gpus'), opts.train.gpus = opts.gpus; end
opts.numSubBatches = 1 ;  % if more than one, must set opts.accumulate=1

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

switch opts.modelType
  case 'sparseNet'
    net = sparseNet_stl10_init('networkType', opts.networkType, 'batchSize', opts.batchSize) ;
  otherwise
    error('Unknown model type ''%s''.', opts.modelType) ;
end

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getSTL10Imdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = imdb.meta.classes(:)' ;
augData = zeros(size(imdb.images.data) + [sum(opts.border(1:2)) ...
  sum(opts.border(3:4)) 0 0], 'like', imdb.images.data); 
augData(opts.border(1)+1:end-opts.border(2), ...
  opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.data; 
imdb.images.augData = augData; 

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3), ...
  'numSlice', opts.numSlice) ;
% getSimpleNNBatch
% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
% images = imdb.images.data(:,:,:,batch) ;
% labels = imdb.images.labels(1,batch) ;
% if rand > 0.5, images=fliplr(images) ; end   %data augmentation
if imdb.images.set(batch(1))==1  % training
  sz0 = size(imdb.images.augData);
  sz = size(imdb.images.data);
  loc = [randi(sz0(1)-sz(1)+1) randi(sz0(2)-sz(2)+1)];
  images = imdb.images.augData(loc(1):loc(1)+sz(1)-1, ...
    loc(2):loc(2)+sz(2)-1, :, batch); 
    if rand > 0.5, images=fliplr(images) ; end
else                              % validating / testing
  images = imdb.images.data(:,:,:,batch); 
end
labels = imdb.images.labels(1,batch) ;


% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% -------------------------------------------------------------------------
function imdb = getSTL10Imdb(opts)
% % -------------------------------------------------------------------------
% % Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'stl10_matlab');
files = [{'train.mat'}, {'test.mat'} ];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
set = [ones(1, 5000), 3*ones(1, 8000)];

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://ai.stanford.edu/~acoates/stl10/stl10_matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end



train_data = load(files{1}, 'X');
train_data = single(train_data.X);
train_labels = (load(files{1}, 'y'));
train_labels = single(train_labels.y);

test_data = (load(files{2}, 'X'));
test_data = single(test_data.X);
test_labels = (load(files{2}, 'y'));
test_labels = single(test_labels.y);


data = [train_data; test_data]';
data = reshape(data, [96, 96, 3, 13000]);

labels = [train_labels; test_labels];
labels = labels';


label_names = load(files{1}, 'class_names');
label_names = label_names.class_names;

% 
% data = cell(1, numel(files));
% labels = cell(1, numel(files));
% sets = cell(1, numel(files));
% for fi = 1:numel(files)
%   fd = load(files{fi}) ;
%   data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
%   labels{fi} = fd.labels' + 1; % Index from 1
%   sets{fi} = repmat(file_set(fi), size(labels{fi}));
% end
% 
% set = cat(2, sets{:});
% data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(mean(mean(data(:,:,:,set == 1), 4), 1), 2);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
  z = reshape(data,[],60000) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
end

if opts.whitenData
  z = reshape(data,[],60000) ;
  W = z(:,set == 1)*z(:,set == 1)'/60000 ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

% clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = labels;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = label_names;
