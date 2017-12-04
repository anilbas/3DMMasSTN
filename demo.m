% This demo shows how the customised layers work on 3DMM as STN.
% It is a simple practical based on gradient descent that demonstrates the
% localiser part of the network: 
% Repeat until converge( x = x - stepsize*grad(f(x))). 
%
% The localiser structure is as follows:
%
% Theta
%   |      
% Split      r               logs         t
%  a|    R = r2R(r)       s = exp(logs)             idx
% Model -> Rotate -> Proj -> Scale -> Translate -> Select -> Euclidean Loss
%
% Test data:
% Images: 224 x 224 x 3 x nbatch 
% Labels: 1 x 2 x 21 x nbatch
%
% Model:
% Check the GitHub page to see how to create the resampled expression
% model.
%
% Dec 2017 || https://github.com/anilbas/3DMMasSTN

%% Load model, data and landmarks:
addpath(genpath(pwd));
model = load('model.mat'); 
load('util/demodata.mat');
idx = readLandmarks('util/landmarks/Landmarks21_112.anl');
nbatch = size(Images,4);
Vis = ones(1,1,21,nbatch, 'single');

% Test - Display
% id=4;
% im = Images(:,:,:,id)./255;
% xp = squeeze(Labels(1,:,:,id));
% vis = Vis(:,id);
% figure; imshow(im); hold on;
% plot(xp(1,:),225-xp(2,:),'xb');
% text(double(xp(1,:)),225-double(xp(2,:)),cellstr(num2str([1:length(xp)]')),'Color','r');

%% Gradient Descent

% Define Initial values for X (6 pose and 10 shape params): 1 x 1 x 16 x nbatch
X = randn(1,1,16,nbatch,'single');
X(:,:,1:3,:)=0;
X(:,:,4:5,:)=112;
X(:,:,6,:)=1;

figure;
for i=1:nbatch
subplot(nbatch,1,i); imshow(Images(:,:,:,i)./255); hold on;
end

% Step size
epsilon = 1e-6; 

% Loop until the loss difference is smaller than 1
diffloss=Inf;
loss=Inf;
while ~(diffloss<1)
    
    % Forward pass
    [alpha,r,t,logs] = vl_nnsplit(X);
       
    X1 = vl_nnmodel(alpha,model);
        
    R = vl_nnr2R(r);
    
    X2 = vl_nnrotate3D(X1,R);
    
    X3 = vl_nnprojection(X2);
    
    s = vl_nnlogScale2Scale(logs);
   
    X4 = vl_nnscale2D(X3,s);
    
    X5 = vl_nntranslate2D(X4,t);
    
    X6 = vl_nnselection(X5,idx);
    % Forward pass-end
    
    % Loss
    preloss=loss;
    loss = vl_nneuclideanloss(X6,Labels,Vis);
    dx6 =  vl_nneuclideanloss(X6,Labels,Vis,1);
    diffloss = norm(loss-preloss);
    % Loss-end
    
    % Display
    delete(findobj('type','line'));
    for i=1:nbatch
        xp = squeeze(X6(1,:,:,i));
        gt = squeeze(Labels(1,:,:,i));
        subplot(nbatch,1,i);
        plot( xp(1,:), 225-xp(2,:), 'r.', gt(1,:), 225-gt(2,:), 'go'); axis equal
    end
    drawnow
    % Display-end
    
    % Backward pass
    dx5 = vl_nnselection(X5,idx,dx6);
    
    [dx4,dt] = vl_nntranslate2D(X4,t,dx5);
    
    [dx3,ds] = vl_nnscale2D(X3,s,dx4);
    
    dlogs = vl_nnlogScale2Scale(logs,ds);
    
    dx2 = vl_nnprojection(X2,dx3); 
    
    [dx1,dR] = vl_nnrotate3D(X1,R,dx2);
    
    dr = vl_nnr2R(r,dR);
    
    dalpha = vl_nnmodel(alpha,model,dx1);
    % Increase the learning rate of the shape parameters
    dalpha = dalpha + 100*dalpha;
    
    dx = vl_nnsplit(X,dalpha,dr,dt,dlogs);
    % Backward pass-end
    
    % Gradient descent
    X = X - epsilon*dx;
    
    disp(num2str(diffloss));
end

%% Display the sampled images using the final values of the localiser (only forward)

grid =  vl_nngrid(X5);
sampler = vl_nnbilinearsampler(Images,grid);

vismask = vl_nnvisibilitymask(X2,model.faces);
vissampler = vl_nnvisibility(sampler,vismask);

figure; 
for i=1:nbatch
subplot(nbatch,5,5*i-4); imshow(Images(:,:,:,i)./255);
subplot(nbatch,5,5*i-3); 
vertices = squeeze(X5(1,:,:,i));
imshow(Images(:,:,:,i)./255); hold on;
plot(vertices(1,:),size(Images(:,:,:,i),2)+1 -vertices(2,:),'.');
subplot(nbatch,5,5*i-2); imshow(sampler(:,:,:,i)./255);
subplot(nbatch,5,5*i-1); imshow(vismask(:,:,:,i));
subplot(nbatch,5,5*i); imshow(vissampler(:,:,:,i)./255);
end
