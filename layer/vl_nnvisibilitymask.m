function [ y ] = vl_nnvisibilitymask( X,faces,dzdy )
% VisibilityMaskLayer
% X	: 1 x 3 x n x b (Rotated vertices)
% faces : nfaces x 3 (Triangles)
% y : 112 x 112 x 3 x b(visibility mask)

nbatch = size(X,4);

if nargin<3
    %forward
    grid_dim = sqrt(size(X, 3));
    y = zeros(grid_dim,grid_dim,3,nbatch,'single');
    FV.faces = faces;
    
    for i= 1:nbatch

        FV.vertices = squeeze(X(1,:,:,i))'; % n x 3  
        mask  =  visibilityNotSelf(FV);
        y(:,:,:,i) = fliplr(imrotate(  single(repmat( reshape(mask,grid_dim,grid_dim)  ,[1 1 3]))   ,-90));
        %single(repmat( reshape(mask,112,112)  ,[1 1 3]))
    end
    
else
    %backward
    y= zeros(size(X),'single');    
end

end

