function [net, info] = dagnn_3dmmasstn(imdb,varargin)

run(fullfile(fileparts(mfilename('fullpath')), ...
    '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.networkType = 'dagnn' ;
opts.derOutputs = {'objective1',0.8998,'objective2',0.1,'objective3',0.0001,'objective4',0.0001};

% expDir: Output directory for the net-epoch-* files and the train.pdf figure
opts.expDir  = fullfile(vl_rootnn, 'examples', '3DMMasSTN', 'data') ;
% dataDir: The VGG directory
opts.dataDir = fullfile(vl_rootnn, 'data', 'models') ;
% The imdb.mat file
opts.imdbPath = fullfile(vl_rootnn, 'data', 'imdb.mat');

opts.theta_learningRate = [4 8];
opts.thetab_weightDecay = 0;

opts.learningRate = 1e-10;
opts.batchSize = 32;
opts.numEpochs = 1000;

opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;

if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------
addpath(genpath(pwd));
% load landmarks
idx = readLandmarks('util/landmarks/Landmarks21_112.anl');
% load model
model = load('model.mat');
% load network
net = dagnn_3dmmasstn_init(model,idx,opts);
% load data
if (~exist('imdb', 'var') || isempty(imdb)), imdb = load(opts.imdbPath); end
% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------
[net, info] = cnn_train_dag(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'derOutputs',opts.derOutputs, ...
    'val', find(imdb.images.set == 2));
% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.train.gpus));
fn = @(x,y) getDagNNBatch(bopts,x,y);
% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch);
labels = imdb.images.labels(:,:,:,batch);

[images, labels] = refineData(images, labels);


if opts.numGpus > 0
    images = gpuArray(images);
    labels = gpuArray(labels);
end

inputs = {'input', images, 'label', labels};
% -------------------------------------------------------------------------
function [Images, Labels] =  refineData(images, labels)
% -------------------------------------------------------------------------
batchSize = size(images,4);

Images = zeros(224,224,3,batchSize*2,'single');
Labels = zeros(1,3,21,batchSize*2,'single');


for i=1:batchSize
    
    id = 2*(i-1)+1;
    
    im = images(:,:,:,i);
    xp = squeeze(labels(:,1:2,:,i));
    vis = squeeze(labels(:,3,:,i));
    
    flippedxp = xp;
    flippedxp(1,:) = ( size(im,2)+1-xp(1,:) ) .*(xp(1,:)~=0);
    flippedxp =  syncFlippedLandmarks( flippedxp );
    
    Images(:,:,:,id) = im;
    Labels(1,1:2,:,id) = xp;
    Labels(1,3,:,id) = vis;
    
    Images(:,:,:,id+1) = fliplr(im);
    Labels(1,1:2,:,id+1) = flippedxp;
    Labels(1,3,:,id+1) = syncFlippedLandmarks(vis);
    
end
