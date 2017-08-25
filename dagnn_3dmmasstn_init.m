function net = dagnn_3dmmasstn_init(model,idx,opts)

%init the network:
net = load([opts.dataDir, '/vgg-face.mat']);
net.layers=net.layers(1:end-2); % removes last fc and softmax layers, so the last layer in the net now is the relu

weightsandbias = xavier(1,1,4096,16);
weightsandbias{1} = weightsandbias{1}.*0.001;
weightsandbias{2}(4:5)=112;
net.layers{end+1} = struct( 'name', 'theta', ...
    'type', 'conv', ...
    'weights', {weightsandbias}, ...
    'stride', 1, ...
    'pad', 0, ...
    'learningRate', opts.theta_learningRate) ;

net = vl_simplenn_tidy(net) ;
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

thetab_idx = net.getParamIndex('thetab');
net.params(thetab_idx).weightDecay = opts.thetab_weightDecay;

%Split layer
splitlayer = split();
net.addLayer('spl', splitlayer,{'x36'},{'alpha','r','t','logs'});

%3D model layer.
model3Dlayer = model3D('model',model);
net.addLayer('mod', model3Dlayer,{'alpha'},{'x38'});

%r2R layer
r2RLayer = r2R();
net.addLayer('r2R', r2RLayer,{'r'},{'R'});

%rotation layer
rotate3DLayer = rotate3D();
net.addLayer('rot', rotate3DLayer,{'x38','R'},{'x39'});

%projection layer
projectionLayer = projection();
net.addLayer('proj', projectionLayer,{'x39'},{'x40'});

%logScale2Scale layer
logScale2ScaleLayer = logScale2Scale();
net.addLayer('logscal', logScale2ScaleLayer,{'logs'},{'s'});

%scale layer
scale2DLayer = scale2D();
net.addLayer('scal', scale2DLayer,{'x40','s'},{'x41'});

%translation layer
translate2DLayer = translate2D();
net.addLayer('tran', translate2DLayer,{'x41','t'},{'x42'});

%selection layer
selectionLayer = selection('idx',idx);
net.addLayer('sel', selectionLayer,{'x42'},{'pred'});

%euclidean loss
euclideanLayer = euclidean();
net.addLayer('euc', euclideanLayer,{'pred','label'},{'objective1'});

%alpha prior loss
sseLayer = sse();
net.addLayer('sse', sseLayer,{'alpha'},{'objective2'});

%%%

%grid layer
gridLayer = resamplegrid();
net.addLayer('grid', gridLayer,{'x42'},{'x43'});

%BilinearSampler layer
BilinearSamplerLayer = dagnn.BilinearSampler();
net.addLayer('samp', BilinearSamplerLayer,{'input','x43'},{'x44'});

%visibilitymask Layer
visibilityMaskLayer = visibilitymask('faces',model.faces);
net.addLayer('mas', visibilityMaskLayer,{'x39'},{'mask'});

%visibility layer
visibilityLayer = visibility();
net.addLayer('visib', visibilityLayer,{'x44','mask'},{'predgrid'});

%siamese loss
siameseLayer = siamese();
net.addLayer('siam', siameseLayer,{'predgrid'},{'objective3'});

%symmetry loss
symmetryLayer = symmetry();
net.addLayer('sym', symmetryLayer,{'predgrid'},{'objective4'});

net.rebuild();
net.meta.inputSize = [224 224 3];
net.conserveMemory = true;
net.meta.trainOpts.learningRate = opts.learningRate;
net.meta.trainOpts.batchSize = opts.batchSize;
net.meta.trainOpts.numEpochs = opts.numEpochs;
end
