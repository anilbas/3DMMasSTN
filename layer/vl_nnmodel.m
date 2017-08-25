function [ y ] = vl_nnmodel( X, model,dzdy )
% ModelLayer
% X     : 1 x 1 x ndims x b
% model : 3 x n
% y     : 1 x 1 x n x b

ndims = size(X,3);
nbatch = size(X,4);

if nargin<3
    %forward
    y = zeros(1,3,model.nverts,nbatch,'single');
    for i= 1:nbatch
        
        y(1,:,:,i) = reshape(model.shapeMU + model.shapePC * squeeze(X(1,1,:,i)),3,model.nverts);
        
    end
    
else
    %backward
     dzdy = reshape(dzdy, 1,size(dzdy,2)*size(dzdy,3),nbatch);

     y = zeros(ndims,nbatch,'single');
     for i= 1:nbatch
     y(:,i) = model.shapePC'*squeeze(dzdy(:,:,i))';
     end
     y = reshape(y,1,1,ndims,nbatch);

end

end


